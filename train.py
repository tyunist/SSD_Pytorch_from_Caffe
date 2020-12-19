from __future__ import division

from models.prototxt_to_pytorch import CaffeNet
from utils.tensorboard_writers import PytorchTBWriter
from utils.logger import get_logger
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.train_saver import Saver
from utils.lr_scheduler import LR_Scheduler
#from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pdb 
import matplotlib.pyplot as plt
import tensorflow as tf
import logging 

TRAIN_MODEL_DEF  =os.environ['TRAIN_MODEL_DEF']
DATA_CONFIG  = os.environ['DATA_CONFIG']
DATASET      = os.environ['DATASET']
LOG_DIR_ROOT = os.environ['LOGDIR']
CROP_IMG_SIZE= os.environ['CROP_IMG_SIZE']
PRETRAINED_CKPT= os.environ['PRETRAINED_CKPT']

class Trainer(object):
    def __init__(self, **kwargs):
        # Fix random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        opt = kwargs.get("args")
        self.opt = opt 
        self.batch_size  = opt.batch_size
        self.img_size    = opt.img_size
        self.epochs      = opt.epochs
        self.start_epoch = opt.start_epoch
        self.ckpt_saver  = Saver(opt)
        
        # Save parameters
        self.ckpt_saver.save_experiment_config({'args': opt})
        print("====================================================")
        print(">> Save params, weights to %s"%self.ckpt_saver.experiment_dir)

        self.tboard_writer = PytorchTBWriter(opt, opt.log_dir)
        print(">> Save tboard logs to %s"%self.tboard_writer.log_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get data configuration
        data_config = parse_data_config(opt.data_config)
        train_path = data_config["train"]
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])
        self.num_classes = len(class_names)
        # Colors for tensorboard visualization    
        self.tf_bbox_colors = np.random.randn(self.num_classes,3)
        
        # Get dataloader
        train_dataset = ListDataset(train_path, img_size=opt.img_size, augment=True,\
                                        multiscale=opt.multiscale_training,\
                                        square_make_type=opt.square_make_type)
        valid_dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, \
                                        multiscale=False,\
                                        square_make_type=opt.square_make_type)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=valid_dataset.collate_fn,
        )
   
        # Define model 
        self.model = CaffeNet(args.caffe_model_def).to(self.device) 
        self.model.apply(weights_init_normal)
        #print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-4)            

        # LR scheduler
        lr_step = 5
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_dataloader), lr_step=lr_step)

        #TODO: load the pretrained ckpt. Select the one given to args.resume first. If not, use one 
        # given in args.pretrained_weights
        pretrained_weights = self.opt.resume
        is_compare_with_saved_RMSE  = True # save a model only if it beats the RMSE of the pretrained model
        if not pretrained_weights:
            pretrained_weights = self.opt.pretrained_weights 
            is_compare_with_saved_RMSE  = False 
        self.best_RMSE = 1e9
        if pretrained_weights:
            ckpt = torch.load(pretrained_weights)
            try:
                if 'state_dict' in ckpt.keys():
                    self.model.load_state_dict(ckpt['state_dict'])
                    print("=> Loaded ckpt %s using ckpt['state_dict']!"%(pretrained_weights))            
                else:
                    self.model.load_state_dict(ckpt)
                    print("=> Loaded ckpt %s using ckpt!"%(pretrained_weights))            
                #self.optimizer.load_state_dict(ckpt['optimizer'])
                self.start_epoch = ckpt['epoch']
                if is_compare_with_saved_RMSE:
                    self.best_RMSE = ckpt['best_RMSE']
                    print("=> Resume ckpt %s, epoch %d, loss = %.4f"%(pretrained_weights, self.start_epoch, self.best_RMSE))            
                else:
                    print("=> Finetune ckpt %s, epoch %d"%(pretrained_weights, self.start_epoch))            
                
            except:
                print("=> Loading ckpt %s using ckpt['state_dict']/ckpt NOT successful!"%(pretrained_weights))            
                pass

        # Set output
        self.model.set_verbose(False) # Turn of forward ... printing 
        self.model.set_train_outputs('mbox_loss','mbox_conf', 'mbox_loc', 'detection_out')
        self.model.set_eval_outputs('mbox_loss','mbox_conf', 'mbox_loc', 'detection_out')
        self.model.set_forward_net_only(True) # Ignore Data, Annotated Data layer if exist
        
        # Detect autograd anomaly
        torch.autograd.set_detect_anomaly(True)
        
        # Save model 
        self.save_ckpt_path = LOG_DIR_ROOT + "/checkpoints/best_model.pth"
    
    def training(self, epoch):
        self.model.train()
        num_batches = len(self.train_dataloader)
        for batch_i, (_, imgs, targets) in enumerate(self.train_dataloader):
            # Note that targets is (N*num_boxes) x 6 where 
            #   targets[i, 0] is the batch index
            #   targets[i, 1] is the object id index (starting from 1)
            #   targets[i, 2:6] is the object bbox (normalized to [0, 1]), in cx,cy,wh format 
            batches_done = len(self.train_dataloader) * epoch + batch_i 
            self.scheduler(self.optimizer, batch_i, epoch-self.start_epoch, self.best_RMSE)
            self.optimizer.zero_grad()
            
            #assert imgs.shape[2] == self.img_size, "Image size %d is not correct. Must be %d"%(imgs.shape[2], self.img_size)
            # Convert Darknet format to Pascal format (x1, y1, x2, y2) normalized 
            targets = darknet_2_pascal_targets(targets)


            imgs    = Variable(imgs.to(self.device), requires_grad=False)
            targets = Variable(targets.to(self.device), requires_grad=False).to(self.device) 
            
            # Only output detection_output at certain iterations
            if batch_i % self.opt.write_image_interval == 0 and self.opt.write_image_interval > 0:
                self.model.set_detection_output(True) 
                self.model.set_train_outputs('mbox_loss','mbox_conf', 'mbox_loc','detection_out')
            else:
                self.model.set_detection_output(False) 
                self.model.set_train_outputs('mbox_loss','mbox_conf', 'mbox_loc')

            blobs   = self.model(imgs, targets)
            loss, loss_l, loss_c  = self.model.get_loss()
            logstr  = "Batch [%d/%d] |lr %.5f |Train loss: %.4f|Loss_l: %.4f|Loss_c: %.4f"%\
                            (batch_i, num_batches, self.scheduler.get_lr(),\
                            loss.item(), loss_l.item(), loss_c.item())    
            print(logstr)
            logging.info(logstr)

            loss.backward()
            self.optimizer.step()
    
            # Visualize loss
            self.tboard_writer.list_of_scalars_summary(\
                    [
                        ('train/total_loss_iter', loss.item()),\
                        ('train/loss_l_iter', loss_l.item()),\
                        ('train/loss_c_iter', loss_c.item()),\
                        ('train/lr', self.scheduler.get_lr())\
                    ], batches_done)         
            # Visualize image & gt bboxes
            if batch_i % self.opt.write_image_interval == 0 and self.opt.write_image_interval > 0:
                self.visualize_batch(batches_done, imgs, targets, self.tboard_writer)

                # Visualize BBox pred 
                #blobs: Tupble with 'mbox_loss','mbox_conf', 'mbox_loc', 'detection_out' tensors
                detections = blobs[-1]      
                assert detections.shape[1] == 7, "Detection out must be Nx7. This one is Nx%d"%detections.shape[1]
                neg_mask1 = detections[:, 3:] < 0.
                neg_mask2 = detections[:, 3:] > 1.0 
                detections[:, 3:][neg_mask1] = 0.001 
                detections[:, 3:][neg_mask2] = 0.99 
                self.visualize_batch(epoch, imgs, detections, self.tboard_writer, msg='train/pred_iter')
    

    def validating(self, epoch):
        self.model.eval()
        num_batches = len(self.valid_dataloader)
        epoch_val_loss = 0
        epoch_loss_l   = 0
        epoch_loss_c   = 0
        for batch_i, (_, imgs, targets) in enumerate(self.valid_dataloader):
            # Note that targets is (N*num_boxes) x 6 where 
            #   targets[i, 0] is the batch index
            #   targets[i, 1] is the object id index (starting from 1)
            #   targets[i, 2:6] is the object bbox (normalized to [0, 1]), in cx,cy,wh format 
            # Convert Darknet format to Pascal format 
            targets = darknet_2_pascal_targets(targets)
            imgs    = Variable(imgs.to(self.device), requires_grad=False)
            targets = Variable(targets.to(self.device), requires_grad=False).to(self.device) 

            # Turn on and off detection_out since it's very slow
            if batch_i == 1:
                self.model.set_detection_output(True) 
                self.model.set_eval_outputs('mbox_loss','mbox_conf', 'mbox_loc','detection_out')
            else:
                self.model.set_detection_output(False) 
                self.model.set_eval_outputs('mbox_loss','mbox_conf', 'mbox_loc')

            with torch.no_grad():
                blobs   = self.model(imgs, targets)
                loss, loss_l, loss_c = self.model.get_loss()

                epoch_val_loss += loss.item()
                epoch_loss_l   += loss_l.item()
                epoch_loss_c   += loss_c.item()
            
            str2Print="Batch [%d/%d]|Val loss %.4f|Loss_l %.4f|Loss_c %.4f"%\
                (batch_i, num_batches, loss.item(), loss_l.item(), loss_c.item())
            print(str2Print)
            logging.info(str2Print)

            # Visualize image & gt bboxes
            if batch_i == 1:
                self.visualize_batch(epoch, imgs, targets, self.tboard_writer, msg='val/gt_epoch')
                # Get BBox errors
                #blobs: Tupble with 'mbox_loss','mbox_conf', 'mbox_loc', 'detection_out' tensors
                detections = blobs[-1]      
                assert detections.shape[1] == 7, "Detection out must be Nx7. This one is Nx%d"%detections.shape[1]
                neg_mask1 = detections[:, 3:] < 0.
                neg_mask2 = detections[:, 3:] > 1.0 
                detections[:, 3:][neg_mask1] = 0.001 
                detections[:, 3:][neg_mask2] = 0.99 
                #detections = torch.masked_select(detections, mask1)
                #detections = torch.masked_select(detections, mask2)
                self.visualize_batch(epoch, imgs, detections, self.tboard_writer, msg='val/pred_epoch')

        epoch_val_loss /= num_batches
        epoch_loss_l   /= num_batches
        epoch_loss_c   /= num_batches
        str2Print="Epoch [%d/%d]|Val loss %.4f|Loss_l %.4f|Loss_c %.4f"%\
            (epoch, self.epochs, epoch_val_loss, epoch_loss_l, epoch_loss_c)
        print("===============================")
        print(str2Print)
        logging.info("===============================")
        logging.info(str2Print)

        # Visualize loss
        self.tboard_writer.list_of_scalars_summary([('val/total_loss_epoch', epoch_val_loss)], epoch)         
        self.tboard_writer.list_of_scalars_summary([('val/loss_l_epoch', epoch_loss_l)], epoch)         
        self.tboard_writer.list_of_scalars_summary([('val/loss_c_epoch', epoch_loss_c)], epoch)         
            
        # Save model 
        if epoch_val_loss < self.best_RMSE:
            is_best = True
            self.ckpt_saver.save_checkpoint({
                                        'epoch': epoch,
                                         'state_dict': self.model.state_dict(),
                                         'optimizer': self.optimizer.state_dict(),
                                         'best_RMSE': epoch_val_loss,
                                        }, is_best)

            print(">> Successful epoch!")
            print(">> Saving model %s"%self.ckpt_saver.ckpt_path)
            logging.info(">> Saving model %s"%self.ckpt_saver.ckpt_path)
            self.best_RMSE = epoch_val_loss
        
        else:
            print(">> Failed epoch since %.4f > %.4f!"%(epoch_val_loss, self.best_RMSE))
            logging.info(">> Failed epoch since %.4f > %.4f!"%(epoch_val_loss, self.best_RMSE))

    
    def visualize_batch(self, step, imgs, targets, tboard_writer, msg='train/gt', max_imgs=2, max_box_per_class=10):
        '''Visualize ground truth bboxes
        Inputs:
            targets N x 6 
        '''    
        if targets.size(1) == 7:
            targets = torch.cat((targets[:,:2], targets[:,3:]) ,1)
        assert targets.size(1) == 6, "Targets must have dimension of Nx6. This targets size Nx%d"%targets.size(1)

        batch_size = imgs.shape[0]
        start_t  = time.time() 
        max_imgs = min(batch_size, max_imgs) 
        list_draw_imgs = []
        for i in range(max_imgs):
            box_idx = targets[:,0] == i
            labels  = targets[box_idx, 1].detach().cpu().numpy()
            boxes   = targets[box_idx, 2:]
            
            # if max_box_per_class: (boxes are sorted)
            if max_box_per_class:
                box_idx = [] 
                for cl in range(1, self.num_classes):
                    cl_idx = np.where(labels == cl)[0] 
                    count  = min(len(cl_idx), max_box_per_class)
                    # shuffer
                    #sub_cl_idx = np.random.choice(cl_idx, count) 
                    sub_cl_idx = cl_idx[:count]
                    if cl == 1:
                        box_idx = sub_cl_idx.flatten()
                    else:
                        box_idx = np.concatenate([box_idx, sub_cl_idx.flatten()]).flatten()
                labels = labels[box_idx]
                boxes  = boxes[box_idx]

            # Convert boxes to y1x1y2x2 format
            boxes   = xyxy2yxyx(boxes)
            if len(boxes.shape) == 1:
                boxes = boxes.view(1,1,-1)
            elif len(boxes.shape) == 2:
                boxes = boxes.unsqueeze(0)
            np_boxes = boxes.detach().cpu().numpy() 
            img = imgs[i].transpose(0,1).transpose(1,2)
            img = img.unsqueeze(0)
            np_img = img.detach().cpu().numpy() 
            
                    
            colors = np.empty([np_boxes.shape[1], 3])
            for j in range(np_boxes.shape[1]):
                colors[j] = self.tf_bbox_colors[int(labels[j])]
            tf_img = tf.image.draw_bounding_boxes(np_img, np_boxes, colors)
            #with  tboard_writer.as_default():
                #tf.summary.image(msg, tf_img, step)

            draw_img = tf_img.numpy() 
            draw_img = draw_img[0].transpose([2, 0, 1])
            list_draw_imgs.append(draw_img[np.newaxis,...])
        
        stacked_draw_imgs = np.concatenate(list_draw_imgs, 0)
        tboard_writer.add_images(msg, stacked_draw_imgs, step) 
        #print(">> VISUAL TIME: ", time.time() - start_t)
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--GPUs", type=str, default='0', help="GPU ID")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['poly', 'step', 'cos'])
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch (in case retrain/resume)")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--caffe_model_def", type=str, default=TRAIN_MODEL_DEF, help="path to caffe model definition file (prototxt)")
    parser.add_argument("--data_config", type=str, default=DATA_CONFIG, help="path to data config file")
    parser.add_argument("--checkpoint_dir", type=str, default=LOG_DIR_ROOT + '/checkpoints', help="path to data config file")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_ROOT + '/logs', help="path to logdir")
    parser.add_argument("--checkname", type=str, default='', help="Subdir of checkpoints")
    parser.add_argument("--resume", type=str, default='', help="Checkpoints to resume")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model", default=PRETRAINED_CKPT)
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=CROP_IMG_SIZE, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--write_image_interval", type=int, default=100, help="interval writing images to tensorboard")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    parser.add_argument("--square_make_type", default='crop', help="How to make the input image have square shape", choices=['crop', 'pad'])
    parser.add_argument("--conf_thres", default=0.5, help="conf threshold", type=float)
    parser.add_argument("--nms_thres", default=0.5, help="nms threshold", type=float)
    parser.add_argument("--visdom", default='visdom', help="Use visdom to visualize or not", type=str)

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.GPUs:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.GPUs:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
		  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    if args.GPUs:
        cudnn.benchmark = True 

    logging, log_experiment_dir = get_logger(args)
    args.log_dir = log_experiment_dir
    print(args)
    trainer = Trainer(args=args)
   

    ## Make repos
    #os.makedirs(LOG_DIR_ROOT + "/logs", exist_ok=True)
    #os.makedirs(LOG_DIR_ROOT + "/output", exist_ok=True)
    #os.makedirs(LOG_DIR_ROOT + "/checkpoints", exist_ok=True)

    # Start training 
    for  epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        
        trainer.validating(epoch)
    
    
    
