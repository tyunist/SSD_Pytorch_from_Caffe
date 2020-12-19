from __future__ import division

from models.prototxt_to_pytorch import CaffeNet
from utils.tensorboard_writers import PytorchTBWriter
from utils.logger import get_logger
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.train_saver import Saver
from demo_detection import Detector

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

TEST_MODEL_DEF  =os.environ['TEST_MODEL_DEF']
DATA_CONFIG  = os.environ['DATA_CONFIG']
DATASET      = os.environ['DATASET']
LOG_DIR_ROOT = os.environ['LOGDIR']
CROP_IMG_SIZE= os.environ['CROP_IMG_SIZE']
PRETRAINED_CKPT= os.environ['PRETRAINED_CKPT']

# Original input image size
INPUT_H      = os.environ['INPUT_H']
INPUT_W      = os.environ['INPUT_W']

class Tester(Detector):
    def __init__(self, **kwargs):
        super(Tester, self).__init__(**kwargs)
        # Get list dataloader
        test_path = self.opt.test_filelist_file
        test_dataset = ListDataset(test_path, img_size=self.opt.img_size, augment=False,\
                                        multiscale=False,\
                                        square_make_type=self.opt.square_make_type)

        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=0, #self.opt.n_cpu,
            pin_memory=False,
            collate_fn=test_dataset.collate_fn,
        )
   
        # Set output
        self.model.set_verbose(False) # Turn of forward ... printing 
        self.model.set_eval_outputs('mbox_loss','mbox_conf', 'mbox_loc', 'detection_out')
        self.model.set_forward_net_only(True) # Ignore Data, Annotated Data layer if exist
    
    def testing(self):
        self.model.eval()
        num_batches = len(self.test_dataloader)
        avg_inference_time = AverageMeter() 
        epoch_val_loss = 0
        epoch_loss_l   = 0
        epoch_loss_c   = 0
        img_paths = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        labels    = [] # Stores the label of all samples
        sample_metrics = []  # List of tuples (TP, confs, pred) 
        for batch_i, (paths, imgs, targets) in enumerate(self.test_dataloader):
            print(f"\n--------------batch {batch_i}/{num_batches}---------------")
            #  Current targets is (N*num_boxes) x 6 where 
            #   targets[i, 0] is the batch index
            #   targets[i, 1] is the object id index (starting from 1)
            #   targets[i, 2:6] is the object bbox (normalized to [0, 1]), in cx,cy,wh format 

            # Convert Darknet format to Pascal format 
            # targets become [batch_idx, obj_idx, x1, y1, x2, y2]
            targets = darknet_2_pascal_targets(targets)
            imgs    = Variable(imgs.to(self.device), requires_grad=False)
            targets = Variable(targets.to(self.device), requires_grad=False) 

            # Turn on and off detection_out since it's very slow
            self.model.set_detection_output(True) 
            self.model.set_eval_outputs('mbox_loss','mbox_conf', 'mbox_loc', 'detection_out')
            prev_time  = time.time() 

            with torch.no_grad():
                blobs   = self.model(imgs, targets)
                loss, loss_l, loss_c = self.model.get_loss()
                if loss.item() < float('inf'):
                    epoch_val_loss += loss.item()
                    epoch_loss_l   += loss_l.item()
                    epoch_loss_c   += loss_c.item()
            
            infer_time = time.time() - prev_time  
            infer_time_2_display = datetime.timedelta(infer_time)
            avg_inference_time.update(infer_time)
            str2Print="Batch [%d/%d]|Val loss %.4f|Loss_l %.4f|Loss_c %.4f"%\
                (batch_i, num_batches, loss.item(), loss_l.item(), loss_c.item())
            #print(str2Print)
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
            # detections: N x 7, where the second dim: [batch_i, cls idx, score, x1, y1, x2, y2] 

            
            assert detections.shape[1] == 7, "Detection out must be Nx7. This one is Nx%d"%detections.shape[1]
            neg_mask1 = detections[:, 3:] < 0.
            neg_mask2 = detections[:, 3:] > 1.0 
            detections[:, 3:][neg_mask1] = 0.001 
            detections[:, 3:][neg_mask2] = 0.99 
          
            # Evaluate each batch using true positive, scores ...
            #TODO: use iou_threshold given from the file
            labels += targets[:, 1].tolist()
            # Note: targets is already N x 6 with bbox in the x1y1x2y2 format

            #sample_metrics += get_batch_statistics(pred_tensor6, targets, iou_threshold=self.model.mbox_loss.threshold)
            sample_metrics += get_batch_statistics(detections.cpu(), targets.cpu(), iou_threshold=self.model.mbox_loss.threshold)

            # Visualize image & gt bboxes
            self.visualize_batch(batch_i, imgs, targets, self.tboard_writer, msg='test/gt_batch')
            self.visualize_batch(batch_i, imgs, detections, self.tboard_writer, msg='test/pred_batch')
            # Store image paths & the detection visualization
            img_paths.append(paths)
            img_detections.append(detections.cpu())
        
        # Compute mAP
        #pdb.set_trace()
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class  = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        evaluation_metrics = [
                                (f"test_loss", epoch_val_loss/num_batches),
                                (f"test_precision", precision.mean()),
                                (f"test_recall", recall.mean()),
                                (f"test_mAP", AP.mean()),
                                (f"test_f1", f1.mean()),
                            ]
        self.tboard_writer.list_of_scalars_summary(evaluation_metrics, 0)
        
        # Save visualization of detections 
        self.save_detection_visualization(img_paths, img_detections) 

        
        # Losses
        epoch_val_loss /= num_batches
        epoch_loss_l   /= num_batches
        epoch_loss_c   /= num_batches
        str2Print="Test loss %.4f|Loss_l %.4f|Loss_c %.4f"%\
            (epoch_val_loss, epoch_loss_l, epoch_loss_c)
        #print("===============================")
        #print(str2Print)
        logging.info("===============================")
        logging.info(str2Print)
        
        
        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")
        

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--GPUs", type=str, default='0', help="GPU ID")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch (in case retrain/resume)")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--caffe_model_def", type=str, default=TEST_MODEL_DEF, help="path to caffe model definition file (prototxt)")
    parser.add_argument("--data_config", type=str, default=DATA_CONFIG, help="path to data config file")
    parser.add_argument("--test_filelist_file", type=str, default='', help="File contains list of test files")
    parser.add_argument("--test_img_dir", type=str, default='', help="Directory to test images")
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
    tester = Tester(args=args)
   

    ## Make repos
    #os.makedirs(LOG_DIR_ROOT + "/logs", exist_ok=True)
    #os.makedirs(LOG_DIR_ROOT + "/output", exist_ok=True)
    #os.makedirs(LOG_DIR_ROOT + "/checkpoints", exist_ok=True)

    # Start testing 
    tester.testing()
    
    
    
