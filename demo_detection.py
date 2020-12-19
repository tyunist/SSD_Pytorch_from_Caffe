from __future__ import division

from models.prototxt_to_pytorch import CaffeNet
from utils.tensorboard_writers import PytorchTBWriter
from utils.logger import get_logger
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.train_saver import Saver
from utils.bbx import CvDrawBboxes
#from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import cv2 as cv

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

DEPLOY_MODEL_DEF  =os.environ['DEPLOY_MODEL_DEF']
DATA_CONFIG  = os.environ['DATA_CONFIG']
DATASET      = os.environ['DATASET']
LOG_DIR_ROOT = os.environ['LOGDIR']
CROP_IMG_SIZE= os.environ['CROP_IMG_SIZE']
PRETRAINED_CKPT= os.environ['PRETRAINED_CKPT']
# Original input image size
INPUT_H      = os.environ['INPUT_H']
INPUT_W      = os.environ['INPUT_W']

class Detector(object):
    def __init__(self, **kwargs):
        opt = kwargs.get("args")
        # Fix random seed
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        self.opt = opt 
        self.batch_size  = opt.batch_size
        self.img_size    = opt.img_size
        

        self.tboard_writer = PytorchTBWriter(opt, opt.log_dir)
        print(">> Save tboard logs to %s"%self.tboard_writer.log_dir)
        device = 'cuda' if (torch.cuda.is_available() and opt.GPUs) else 'cpu'
        self.device = torch.device(device)

        # Get data configuration
        data_config = parse_data_config(opt.data_config)
        test_path = opt.test_img_dir
        self.class_names = load_classes(data_config["names"])
        self.num_classes = len(self.class_names)
        # Colors for tensorboard visualization    
        self.tf_bbox_colors = np.random.randn(self.num_classes,3)
        
        # Get folder dataloader
        test_dataset = ImageFolder(test_path, img_size=opt.img_size, square_make_type=opt.square_make_type)


        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
            pin_memory=True,
        )
   
        # Define model 
        self.model = CaffeNet(self.opt.caffe_model_def).to(self.device) 
        self.model.apply(weights_init_normal)
        #print(self.model)
        # Load the pretrained ckpt
        pretrained_weights = self.opt.pretrained_weights 
        if not pretrained_weights:
            pretrained_weights = self.opt.resume
        assert pretrained_weights, 'Need to provide pretrained weights via either --pretrained_weights or --resume'
        ckpt = torch.load(pretrained_weights)
        if 'state_dict' in ckpt.keys():
            self.model.load_state_dict(ckpt['state_dict'])
            print("=> Loaded ckpt %s using ckpt['state_dict']!"%(pretrained_weights))            
        else:
            self.model.load_state_dict(ckpt)
            print("=> Loaded ckpt %s using ckpt!"%(pretrained_weights))            

        try:
            self.best_RMSE = ckpt['best_RMSE']
            self.start_epoch = ckpt['epoch']
            print("=> Loaded ckpt %s, epoch %d, loss = %.4f"%(pretrained_weights, self.start_epoch, self.best_RMSE))            
        except:
            print("=> Loaded ckpt %s"%(pretrained_weights))            

        # Set output
        self.model.set_verbose(False) # Turn of forward ... printing 
        self.model.set_eval_outputs('detection_out')
        self.model.set_forward_net_only(True) # Ignore Data, Annotated Data layer if exist

    def get_net_profile(self):
        x = torch.randn((1, 3, self.opt.img_size, self.opt.img_size), requires_grad=True)
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            self.model(x)
        print(prof)
        prof.export_chrome_trace("profile.html")
        
    def detecting(self):
        self.model.eval()
        num_batches = len(self.test_dataloader)
        avg_inference_time = AverageMeter() 
        img_paths = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        for batch_i, (paths, imgs) in enumerate(self.test_dataloader):
            imgs    = Variable(imgs.to(self.device), requires_grad=False)

            # Turn on and off detection_out since it's very slow
            self.model.set_detection_output(True) 
            self.model.set_eval_outputs('detection_out')
            prev_time  = time.time() 

            with torch.no_grad():
                blobs  = self.model(imgs)
            
            infer_time = time.time() - prev_time  
            infer_time_2_display = datetime.timedelta(infer_time)
            avg_inference_time.update(infer_time)

            str2Print="- Batch [%d/%d]| Infer time: %s[%.5f]s"%(batch_i, num_batches, infer_time_2_display,\
                                                                avg_inference_time.avg)
            print(str2Print)
            logging.info(str2Print)
            

            # Visualize image & gt bboxes
            # Get BBox errors
            #blobs: Tupble with 'mbox_loss','mbox_conf', 'mbox_loc', 'detection_out' tensors
            if isinstance(blobs, (list, tuple)):
                detections = blobs[-1]      
            else:
                detections = blobs      
            if len(detections.size()) == 1:
                detections = detections.unsqueeze(0)
            
            assert detections.shape[1] == 7, "Detection out must be Nx7. This one is Nx%d"%detections.shape[1]
            print(">> Number boxes per image: %.2f"%(detections.size(0)/float(self.batch_size)))
            neg_mask1 = detections[:, 3:] < 0.
            neg_mask2 = detections[:, 3:] > 1.0 
            detections[:, 3:][neg_mask1] = 0.001 
            detections[:, 3:][neg_mask2] = 0.99 
            #detections = torch.masked_select(detections, mask1)
            #detections = torch.masked_select(detections, mask2)
            self.visualize_batch(batch_i, imgs, detections, self.tboard_writer, msg='test/pred_batch', max_box_per_class=3)

            # Save image paths and detections
            img_paths.append(paths)
            img_detections.append(detections.cpu())
      
        # Save visualization of detections 
        self.save_detection_visualization(img_paths, img_detections) 


    def save_detection_visualization(self, img_paths, img_detections):
        # Save image detections 
        print("\n==================================\nSaving detections:")
        bbox_drawer = CvDrawBboxes(self.class_names, INPUT_H, INPUT_W)
        out_dir     = os.path.join(self.opt.log_dir,  'detections', self.opt.test_img_dir.split('/')[-1])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("\n>> Save output detections to: %s"%out_dir)

        # Iterate through images and save plot of detections
        for batch_i, (b_paths, b_detections) in enumerate(zip(img_paths, img_detections)):
            for img_i, path in enumerate(b_paths):
                detections = b_detections[b_detections[:,0]==img_i]
                print("(%d) Image: '%s'" % (img_i, path))
    
                img = cv.imread(path) 
                if detections is not None and torch.sum(detections) > 0:
                    # Convert detections of size N x 7, where the last dim: [batch_i, cls idx, score, x1, y1, x2, y2] 
                    # to [x1, y1, x2, y2, conf, cls_conf, cls_pred] for drawing
                    num         = detections.size(0)
                    detections  = torch.cat((self.img_size*detections[:,3:].view(num,4), # x1, y1, x2, y2
                                        torch.ones([num, 1]).cpu(), # fake cls_conf 
                                        detections[:, 2].view(num,1), # conf
                                        detections[:, 1].view(num,1), # cls_pred
                                       ),-1)
                    
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections, self.opt.img_size, img.shape[:2])        
                filename = path.split("/")[-1].split(".")[0]
                savefig_name = f"{out_dir}/{filename}.png"
                bbox_drawer.drawBboxes(img, detections, savefig_name)
                
                print(">> Save detection + img: %s"%(savefig_name)) 

        print("===============================")

    
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--GPUs", type=str, default='0', help="GPU ID")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--caffe_model_def", type=str, default=DEPLOY_MODEL_DEF, help="path to caffe model definition file (prototxt)")
    parser.add_argument("--data_config", type=str, default=DATA_CONFIG, help="path to data config file")
    parser.add_argument("--test_img_dir", type=str, default='', help="path to test images")
    parser.add_argument("--checkpoint_dir", type=str, default=LOG_DIR_ROOT + '/checkpoints', help="path to data config file")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_ROOT + '/logs', help="path to logdir")
    parser.add_argument("--checkname", type=str, default='', help="Subdir of checkpoints")
    parser.add_argument("--resume", type=str, default='', help="Checkpoints to resume")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model", default=PRETRAINED_CKPT)
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=CROP_IMG_SIZE, help="size of each image dimension")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--write_image_interval", type=int, default=500, help="interval writing images to tensorboard")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
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
    
    # Logging
    logging, log_experiment_dir = get_logger(args)
    args.log_dir = log_experiment_dir

    print(args)
    detector = Detector(args=args)
   
    # Start detecting 
    detector.detecting()
    
    # Profiling model
    #detector.get_net_profile()
    
    
    
