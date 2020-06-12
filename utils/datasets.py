import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict, deque
import pdb


def pad_to_square(*inputs):
    '''
    Inputs: 
        img, pad_value in this order
    '''
    pad_value = 0
    if len(inputs) == 2:
        img = inputs[0]
        pad_value = inputs[1]
    elif len(inputs) == 1:
        img = inputs[0]
    else:
        raise TypeError("Input to pad_to_square() must be: img, pad_value or img!")
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def center_crop_to_square(*inputs):
    '''
    Inputs:
        img 
    Outputs:
        cropped square image
        crop: List[crop1, crop2, crop3, crop4]. Crop values are <= 0 
              to differentiate a return from this center_crop_to_square() vs that of pad_to_square()
    '''
    img = inputs[0]

    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    crop1, crop2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    crop = [0, 0, crop1, crop2] if h <  w else [crop1, crop2, 0, 0]
    
    img  = img[:,crop[0]:h-crop[1], crop[2]:w-crop[3]]
   
    # Cropping change only left, upper coordinate 
    #crop = [-c for c in crop]
    crop = [-crop1, -crop2, 0, 0] if h < w else [0,0, -crop1, -crop2]
    return img, crop 


square_make_funcs = {'crop': center_crop_to_square,
                     'pad': pad_to_square}


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, square_make_type='pad'):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        # How to make the image square: two options given in square_make_funcs{}
        self.square_make_type = square_make_type
        self.square_make_func = square_make_funcs[square_make_type]


    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = self.square_make_func(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class OnlineLoader(object):
    def __init__(self, img_size=416, max_buffer_size=20):
        '''buffer_size: size of buffer (in items)'''
        self.img_size = img_size
        self.buffer   = deque(maxlen=max_buffer_size) 
        self.max_buffer_size = max_buffer_size 

    def feedBuffer(self, img_ids, imgs):
        for i, (img_id, img) in enumerate(zip(img_ids, imgs)):
            self.buffer.append([img_id, img])
        self.buffer_size = len(self.buffer)

    def getBatch(self, batch_size):
        imgs     = [] 
        img_ids  = []
        for i in range(batch_size):
            img_id, img = self.getItem()
            imgs.append(img)
            img_ids.append(img_id)
        return img_ids, torch.stack(imgs, dim=0)
        
    

    def getItem(self):
        img_id, img = self.buffer.pop()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Transform image to PyTorch tensor
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        
        self.buffer_size -= 1

        return img_id, img

    def len(self):
        return self.buffer_size


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, square_make_type='pad'):
        '''
        Inputs:
            square_make_type: how to enforce the square, either 'crop' or 'pad'
        '''
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace('Images', 'Labels')
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 1 * 32
        self.max_size = self.img_size + 6 * 32
        self.batch_count = 0
        
        # How to make the image square: two options given in square_make_funcs{}
        self.square_make_type = square_make_type
        self.square_make_func = square_make_funcs[square_make_type]

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip().split(',')[0]
        except:
            img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad/crop to square resolution
        img, pad = self.square_make_func(img, 0) # call it pad but it can be crop 
        #img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
         
        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip().split(',')[0]
        except:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            #TODO: check if labels start from 0 or not. +1 if starting from 0
            if torch.sum(boxes[:, 0] == 0) > 0:
                boxes[:,0] += 1
                
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            ## Returns (cx, cy, w, h)
            #boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            #boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            #boxes[:, 3] *= w_factor / padded_w
            #boxes[:, 4] *= h_factor / padded_h


            #Returns (x1, y1, x2, y2)
            boxes[:, 1] = torch.clamp(x1/float(padded_w), min=0.0, max=1.0)
            boxes[:, 2] = torch.clamp(y1/float(padded_h), min=0.0, max=1.0)
            boxes[:, 3] = torch.clamp(x2/float(padded_w), min=0.0, max=1.0)
            boxes[:, 4] = torch.clamp(y2/float(padded_h), min=0.0, max=1.0)

            # Get rid of invalid boxes which have zero area 
            x2_le_x1_m = boxes[:, 3] < boxes[:,1] + 1e-4    #y1 < x1
            y2_le_y1_m = boxes[:, 4] < boxes[:,2] + 1e-4    #y1 < x1

            neg_mask = x2_le_x1_m + y2_le_y1_m 
            neg_mask = neg_mask > 0 
            pos_mask = neg_mask.logical_not() 

            if pos_mask.sum() == 0:
                targets = None 
            else:
                boxes = boxes[pos_mask]
                # Return targets in cx, cy, w, h shape
                b_only = boxes[:, 1:]
                boxes[:,1:] = torch.cat([(b_only[:, 2:] + b_only[:, :2])/2,  # cx, cy
                                          b_only[:, 2:] - b_only[:, :2]],\
                                       1)  
                if len(boxes.shape) == 1:
                    boxes = boxes.unsqueeze(0)
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes
            

        # Apply augmentations
        if self.augment and targets is not None:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        #imgs = torch.stack([img for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
