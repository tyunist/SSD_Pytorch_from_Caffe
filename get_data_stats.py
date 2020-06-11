from __future__ import division

from utils.logger import *
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
from train import Trainer

DATA_CONFIG  = os.environ['DATA_CONFIG']
DATASET      = os.environ['DATASET']
LOG_DIR_ROOT = os.environ['LOGDIR']
CROP_IMG_SIZE= os.environ['CROP_IMG_SIZE']

class DataStats(Trainer):
    def __init__(self, **kwargs):
        # Fix random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        opt = kwargs.get("args")
        self.opt = opt 
        self.batch_size  = opt.batch_size
        self.img_size    = opt.img_size
        
        self.tboard_summary = PytorchLogger(opt)
        print(">> Save tboard logs to %s"%self.tboard_summary.log_dir)
        
        # Get data configuration
        data_config = parse_data_config(opt.data_config)
        train_path = data_config["train"]
        class_names = load_classes(data_config["names"])
        self.num_classes = len(class_names)
        # Colors for tensorboard visualization    
        self.tf_bbox_colors = np.random.randn(self.num_classes,3)
        
        # Get dataloader
        train_dataset = ListDataset(train_path, img_size=opt.img_size, augment=False,\
                                        multiscale=False)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

    
    def get_img_stats(self):
        num_batches = len(self.train_dataloader)
        rgb_avger   = AverageMeter()
        rgb_stder   = AverageMeter()
        max_epochs  = 5
        for epoch in range(max_epochs):
            for batch_i, (_, imgs, targets) in enumerate(self.train_dataloader):
                # Note that targets is (N*num_boxes) x 6 where 
                #   targets[i, 0] is the batch index
                #   targets[i, 1] is the object id index (starting from 0)
                #   targets[i, 2:6] is the object bbox (normalized to [0, 1]), in cx,cy,wh format 
                # Convert Darknet format to Pascal format (x1, y1, x2, y2) normalized 
                targets = darknet_2_pascal_targets(targets)
                batches_done = len(self.train_dataloader) * epoch + batch_i 


                logstr  = "Batch [%d/%d]"%(batch_i, num_batches)
                print(logstr)
                
                # Get RGB mean, std
                rgb_mean = imgs.mean([0, 2, 3])    
                rgb_std  = imgs.std([0, 2, 3])    
                rgb_avger.update(rgb_mean)                   
                rgb_stder.update(rgb_std)                   
                print(">>Mean, std before standardizing:",rgb_mean,'\n', rgb_std)

                # standardize 
                s_imgs = (imgs - rgb_mean.view(1,3,1,1))/rgb_std.view(1,3,1,1)
                s_rgb_mean = s_imgs.mean([0, 2, 3])    
                s_rgb_std  = s_imgs.std([0, 2, 3])    
                print(">> Mean, std after standardizing:",s_rgb_mean, '\n', s_rgb_std)


                # Visualize image & gt bboxes
                if batch_i % self.opt.write_image_interval == 0 and self.opt.write_image_interval > 0:
                    self.visualize_batch(batches_done, imgs, targets, self.tboard_summary)
            str2log = "\n=============================\n Epoch [%d/%d]"%(epoch, max_epochs)
            str2log.join(["\nMean of mean(RGB):", rgb_avger.avg.cpu().numpy().array_str()]) 
            str2log.join(["\nStd of mean(RGB):", rgb_avger.std.cpu().numpy().array_str()]) 
            str2log.join(["\nMean of std(RGB):", rgb_stder.avg.cpu().numpy().array_str()]) 
            str2log.join(["\nStd of std(RGB):", rgb_stder.std.cpu().numpy().array_str()]) 
            str2log.join("\n\n")
            str2log.join("===============================")
            print(str2log)
            logging.info(str2log)

            # Epoch [4/5]
            #* Mean of RGB means: tensor([0.2338, 0.2387, 0.1948])
            #* Std of RGB means: tensor([0.0075, 0.0086, 0.0089])
            #* Mean of RGB stds: tensor([0.2729, 0.2781, 0.2582])
            #* Std of RGB stds: tensor([0.0072, 0.0075, 0.0103])

    def get_bbox_stats(self):
        num_batches = len(self.train_dataloader)
        for batch_i, (_, imgs, targets) in enumerate(self.train_dataloader):
            # Note that targets is (N*num_boxes) x 6 where 
            #   targets[i, 0] is the batch index
            #   targets[i, 1] is the object id index (starting from 0)
            #   targets[i, 2:6] is the object bbox (normalized to [0, 1]), in cx,cy,wh format 
            # Convert Darknet format to Pascal format (x1, y1, x2, y2) normalized 
            targets = darknet_2_pascal_targets(targets)

            logstr  = "Batch [%d/%d]"%(batch_i, num_batches)
            print(logstr)
            


            # Visualize image & gt bboxes to ensure that everything is correct!
            if batch_i % self.opt.write_image_interval == 0 and self.opt.write_image_interval > 0:
                self.visualize_batch(batches_done, imgs, targets, self.tboard_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--GPUs", type=int, default=0, help="GPU ID")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Name of the dataset")
    parser.add_argument("--stat_type", type=str, default='', help="What type of stastistics to calculate", choices=['bbox_kmean', 'pixel_mean'])
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default=DATA_CONFIG, help="path to data config file")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_ROOT + '/logs', help="path to logdir")
    parser.add_argument("--checkname", type=str, default='', help="Subdir of checkpoints")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=CROP_IMG_SIZE, help="size of each image dimension")
    parser.add_argument("--write_image_interval", type=int, default=500, help="interval writing images to tensorboard")
    parser.add_argument("--visdom", default='visdom', help="Use visdom to visualize or not", type=str)

    args = parser.parse_args()
    print(args)
    data_stats = DataStats(args=args)

    if torch.cuda.is_available():
        if args.GPUs:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.GPUs:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
		  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
   
    if args.stat_type:
        if args.stat_type == 'pixel_mean': 
            data_stats.get_img_stats()
        elif args.stat_type == 'bbox_kmean': 
            data_stats.get_bbox_stats()
    
    else:    
        data_stats.get_img_stats()
        data_stats.get_bbox_stats()
    
