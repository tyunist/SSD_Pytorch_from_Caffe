# Convert SSD Model from Caffe to Pytorch
This tool aims to load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch.
Adopted from [this](https://github.com/marvis/pytorch-caffe)
***
## Prerequisites:
- [ ] Pytorch >= 1.4
- [ ] Python  >= 3.5
We use Python3.7. To use other versions, change python version in script files such as run_demo.sh, run_train.sh

***
## Train a model using the Caffe model's prototxt file  with Pytorch 
(using caffe dataloader style)
### Prepare data 
- [ ] All object should be indexed starting from 1. The number of classes used in the model's prototxt file should be equal to the number of types of objects + 1 (background)
- [ ] The ground truth's input to the network should be a tensor of size B x 6, where 
```
B = \sum_{images in batch} (number of objects in the image)
``` 
The second dimension of the tensor represents a box
```
[batch_index, class_index, x1, y1, x2, y2]
```
Where [x1,y1, x2,y2] are all normalized to [0,1].
 
### Start training
```
bash run_train.sh
```
* Note: we use python3.7. 

***
## Test on images located in a folder
### Get the pretrained networks
Pretrained model can be downloaded from [this link](https://drive.google.com/drive/folders/1fP9DXTmxrna_5vJyOA5pmvUO8uOq08Xn?usp=sharing)

Download and save the pretrained_ckpt folder to the same repo with train.py

### Edit the run_demo.sh 
```
    # Path to the pretrained network 
    --pretrained_weights pretrained_ckpt/best_checkpoint.pth.tar \
    # Path to the test images
    --test_img_dir examples_images/three_drones/test_from_skydio_hd_ian_house \

```
And run 
```
bash run_demo.sh
```
The results are visualized using Tensorboard. Go to the log file location, run tensorboard, open the link
that is shown up to display predicted boxes.

***
## SSD300 Caffe model structure (n=21 classes)
### Todos
- [x] support forward classification networks: AlexNet, VGGNet, GoogleNet, [ResNet](http://pan.baidu.com/s/1kVm4ly3), [ResNeXt](https://pan.baidu.com/s/1pLhk0Zp#list/path=%2F), DenseNet
- [x] support forward detection networks: [SSD300](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA), [S3FD](https://github.com/sfzhang15/SFD), FPN
- [x] Support custom layers from ASU team

### Supported Layers
Each layer in caffe will have a corresponding layer in pytorch. 
- [x] Convolution
- [x] InnerProduct
- [x] BatchNorm
- [x] Scale
- [x] ReLU
- [x] Pooling
- [x] Reshape
- [x] Softmax
- [x] Accuracy
- [x] SoftmaxWithLoss
- [x] Dropout
- [x] Eltwise
- [x] Normalize
- [x] Permute
- [x] Flatten
- [x] Slice
- [x] Concat
- [x] PriorBox
- [x] LRN : gpu version is ok, cpu version produce big difference
- [x] DetectionOutput: support batchsize=1, num_classes=1 forward
- [x] Crop
- [x] Deconvolution
- [x] MultiBoxLoss


 abc | def 
 --- | --- 

### SSD300 Model structure
 
Layer| Input Tensor | Output Tensor
---   |         ---    |        ---
forward conv1_2                       | [8, 64, 300, 300] -> |[8, 64, 300, 300]    
 forward relu1_2                       | [8, 64, 300, 300] -> |[8, 64, 300, 300]
forward pool1                         | [8, 64, 300, 300] -> |[8, 64, 150, 150]
forward conv2_1                       | [8, 64, 150, 150] -> |[8, 128, 150, 150]
forward relu2_1                       | [8, 128, 150, 150] ->| [8, 128, 150, 150]
forward conv2_2                       | [8, 128, 150, 150] ->| [8, 128, 150, 150]
forward relu2_2                       | [8, 128, 150, 150] ->| [8, 128, 150, 150]
forward pool2                         | [8, 128, 150, 150] ->| [8, 128, 75, 75]
forward conv3_1                       | [8, 128, 75, 75] -> |[8, 256, 75, 75]
forward relu3_1                       | [8, 256, 75, 75] -> |[8, 256, 75, 75]
forward conv3_2                       | [8, 256, 75, 75] -> |[8, 256, 75, 75]
forward relu3_2                       | [8, 256, 75, 75] -> |[8, 256, 75, 75]
forward conv3_3                       | [8, 256, 75, 75] -> |[8, 256, 75, 75]
forward relu3_3                       | [8, 256, 75, 75] -> |[8, 256, 75, 75]
forward pool3                         | [8, 256, 75, 75] -> |[8, 256, 38, 38]
forward conv4_1                       | [8, 256, 38, 38] -> |[8, 512, 38, 38]
forward relu4_1                       | [8, 512, 38, 38] -> |[8, 512, 38, 38]
forward conv4_2                       | [8, 512, 38, 38] -> |[8, 512, 38, 38]
forward relu4_2                       | [8, 512, 38, 38] -> |[8, 512, 38, 38]
forward conv4_3                       | [8, 512, 38, 38] -> |[8, 512, 38, 38]
forward relu4_3                       | [8, 512, 38, 38] -> |[8, 512, 38, 38]
forward pool4                         | [8, 512, 38, 38] -> |[8, 512, 19, 19]
forward conv5_2                       | [8, 512, 19, 19] -> |[8, 512, 19, 19]
forward relu5_2                       | [8, 512, 19, 19] -> |[8, 512, 19, 19]
forward conv5_3                       | [8, 512, 19, 19] -> |[8, 512, 19, 19]
forward relu5_3                       | [8, 512, 19, 19] -> |[8, 512, 19, 19]
forward pool5                         | [8, 512, 19, 19] -> |[8, 512, 19, 19]
forward fc6                           | [8, 512, 19, 19] -> |[8, 1024, 19, 19]
forward relu6                         | [8, 1024, 19, 19] ->| [8, 1024, 19, 19]
forward fc7                           | [8, 1024, 19, 19] ->| [8, 1024, 19, 19]
forward relu7                         | [8, 1024, 19, 19] ->| [8, 1024, 19, 19]
forward conv6_1                       | [8, 1024, 19, 19] ->| [8, 256, 19, 19]
forward conv8_1_relu                  | [8, 128, 5, 5] -> |[8, 128, 5, 5]
forward conv8_2                       | [8, 128, 5, 5] -> |[8, 256, 3, 3]
forward conv8_2_relu                  | [8, 256, 3, 3] -> |[8, 256, 3, 3]
forward conv9_1                       | [8, 256, 3, 3] -> |[8, 128, 3, 3]
forward conv9_1_relu                  | [8, 128, 3, 3] -> |[8, 128, 3, 3]
forward conv9_2                       | [8, 128, 3, 3] -> |[8, 256, 1, 1]
forward conv9_2_relu                  | [8, 256, 1, 1] -> |[8, 256, 1, 1]
forward conv4_3_norm                  | [8, 512, 38, 38] ->| [8, 512, 38, 38]
forward conv4_3_norm_mbox_loc         | [8, 512, 38, 38] ->| [8, 16, 38, 38]
forward conv4_3_norm_mbox_loc_perm    | [8, 16, 38, 38] -> | [8, 38, 38, 16]
forward conv4_3_norm_mbox_loc_flat    | [8, 38, 38, 16] -> | [8, 23104]
forward conv4_3_norm_mbox_conf        | [8, 512, 38, 38] ->| [8, 84, 38, 38]
forward conv4_3_norm_mbox_conf_perm   | [8, 84, 38, 38] -> | [8, 38, 38, 84]
forward conv4_3_norm_mbox_conf_flat   | [8, 38, 38, 84] -> | [8, 121296]
forward conv4_3_norm_mbox_priorbox    | [8, 512, 38, 38] ->| [1, 2, 23104]
forward fc7_mbox_loc                  | [8, 1024, 19, 19]->| [8, 24, 19, 19]
forward fc7_mbox_loc_perm             | [8, 24, 19, 19] -> | [8, 19, 19, 24]
forward fc7_mbox_loc_flat             | [8, 19, 19, 24] -> | [8, 8664]
forward fc7_mbox_conf                 | [8, 1024, 19, 19]->| [8, 126, 19, 19]
forward fc7_mbox_conf_perm            | [8, 126, 19, 19] ->| [8, 19, 19, 126]
forward fc7_mbox_conf_flat            | [8, 19, 19, 126] ->| [8, 45486]
forward fc7_mbox_priorbox             | [8, 1024, 19, 19]->| [1, 2, 8664]
forward conv6_2_mbox_loc              | [8, 512, 10, 10] -> |[8, 24, 10, 10]
forward conv6_2_mbox_loc_perm         | [8, 24, 10, 10] ->  |[8, 10, 10, 24]
forward conv6_2_mbox_loc_flat         | [8, 10, 10, 24] ->  |[8, 2400]
forward conv6_2_mbox_conf             | [8, 512, 10, 10] -> |[8, 126, 10, 10]
forward conv6_2_mbox_conf_perm        | [8, 126, 10, 10] -> |[8, 10, 10, 126]
forward conv6_2_mbox_conf_flat        | [8, 10, 10, 126] -> |[8, 12600]
forward conv6_2_mbox_priorbox         | [8, 512, 10, 10] -> |[1, 2, 2400]
forward conv7_2_mbox_loc              | [8, 256, 5, 5] ->   |[8, 24, 5, 5]
forward conv7_2_mbox_loc_perm         | [8, 24, 5, 5] ->    |[8, 5, 5, 24]
forward conv7_2_mbox_loc_flat         | [8, 5, 5, 24] ->    |[8, 600]
forward conv7_2_mbox_conf             | [8, 256, 5, 5] ->| [8, 126, 5, 5]
forward conv7_2_mbox_conf_perm        | [8, 126, 5, 5] ->| [8, 5, 5, 126]
forward conv7_2_mbox_conf_flat        | [8, 5, 5, 126] ->| [8, 3150]
forward conv7_2_mbox_priorbox         | [8, 256, 5, 5] ->| [1, 2, 600]
forward conv8_2_mbox_loc              | [8, 256, 3, 3] ->| [8, 16, 3, 3]
forward conv8_2_mbox_loc_perm         | [8, 16, 3, 3] -> | [8, 3, 3, 16]
forward conv8_2_mbox_loc_flat         | [8, 3, 3, 16] -> | [8, 144]
forward conv8_2_mbox_conf             | [8, 256, 3, 3] ->| [8, 84, 3, 3]
forward conv8_2_mbox_conf_perm        | [8, 84, 3, 3] -> | [8, 3, 3, 84]
forward conv8_2_mbox_conf_flat        | [8, 3, 3, 84] -> | [8, 756]
forward conv8_2_mbox_priorbox         | [8, 256, 3, 3] ->| [1, 2, 144]
forward conv9_2_mbox_loc              | [8, 256, 1, 1] ->| [8, 16, 1, 1]
forward conv9_2_mbox_loc_perm         | [8, 16, 1, 1] -> | [8, 1, 1, 16]
forward conv9_2_mbox_loc_flat         | [8, 1, 1, 16] -> | [8, 16]
forward conv9_2_mbox_conf             | [8, 256, 1, 1] ->| [8, 84, 1, 1]
forward conv9_2_mbox_conf_perm        | [8, 84, 1, 1] -> | [8, 1, 1, 84]
forward conv9_2_mbox_conf_flat        | [8, 1, 1, 84] -> | [8, 84]
forward conv9_2_mbox_priorbox         | [8, 256, 1, 1] ->| [1, 2, 16]
forward mbox_loc                      | [8, 23104] ->    | [8, 34928]
forward mbox_conf                     | [8, 121296] ->   | [8, 183372]
forward mbox_priorbox                 | [1, 2, 23104] -> | [1, 2, 34928]
 

#### mbox_conf is the concatenation of: 
                                                                   
Layer Name  | Input Tensor | Output Tensor
---     |     ---      |    ---
forward conv4_3_norm_mbox_conf       |  [8, 512, 38, 38] -> | [8, 84, 38, 38]	
forward conv4_3_norm_mbox_conf_perm  |  [8, 84, 38, 38] ->  | [8, 38, 38, 84]	
forward conv4_3_norm_mbox_conf_flat  |  [8, 38, 38, 84] ->  | [8, 121296]     
forward fc7_mbox_conf                |  [8, 1024, 19, 19] ->| [8, 126, 19, 19]	
forward fc7_mbox_conf_perm           |  [8, 126, 19, 19] -> | [8, 19, 19, 126]	
forward fc7_mbox_conf_flat           |  [8, 19, 19, 126] -> | [8, 45486]
forward conv6_2_mbox_conf            |  [8, 512, 10, 10] -> | [8, 126, 10, 10]	
forward conv6_2_mbox_conf_perm       |  [8, 126, 10, 10] -> | [8, 10, 10, 126]	
forward conv6_2_mbox_conf_flat       |  [8, 10, 10, 126] -> | [8, 12600]  
forward conv7_2_mbox_conf            |  [8, 256, 5, 5] ->   | [8, 126, 5, 5]	
forward conv7_2_mbox_conf_perm       |  [8, 126, 5, 5] ->   | [8, 5, 5, 126]	
forward conv7_2_mbox_conf_flat       |  [8, 5, 5, 126] ->   | [8, 3150]
forward conv8_2_mbox_conf            |  [8, 256, 3, 3] ->   | [8, 84, 3, 3]	
forward conv8_2_mbox_conf_perm       |  [8, 84, 3, 3] ->    | [8, 3, 3, 84]	
forward conv8_2_mbox_conf_flat       |  [8, 3, 3, 84] ->    | [8, 756]
forward conv9_2_mbox_conf            |  [8, 256, 1, 1] ->   | [8, 84, 1, 1]	
forward conv9_2_mbox_conf_perm       |  [8, 84, 1, 1] ->    | [8, 1, 1, 84]	
forward conv9_2_mbox_conf_flat       |  [8, 1, 1, 84] ->    | [8, 84]

Concatenation: 121296 + 45486 + 12600 + 3150 + 756 + 84 = 183372 
All of these have taken into account 21 classes. 
In order to work with three_drones dataset which has only 3 classes, we have to change the number of output channels of every element in this concatenation by 7


#### mbox_loc is the concatenation of: 
Layer Name  | Input Tensor | Output Tensor
---     |     ---      |    ---
forward conv4_3_norm_mbox_loc        |  [8, 512, 38, 38] ->| [8, 16, 38, 38]
forward conv4_3_norm_mbox_loc_perm   |  [8, 16, 38, 38] -> | [8, 38, 38, 16]
forward conv4_3_norm_mbox_loc_flat   |  [8, 38, 38, 16] -> | [8, 23104]
forward fc7_mbox_loc                 |  [8, 1024, 19, 19]->| [8, 24, 19, 19]
forward fc7_mbox_loc_perm            |  [8, 24, 19, 19] -> | [8, 19, 19, 24]
forward fc7_mbox_loc_flat            |  [8, 19, 19, 24] -> | [8, 8664]
forward conv6_2_mbox_loc             |  [8, 512, 10, 10] ->| [8, 24, 10, 10]
forward conv6_2_mbox_loc_perm        |  [8, 24, 10, 10] -> | [8, 10, 10, 24]
forward conv6_2_mbox_loc_flat        |  [8, 10, 10, 24] -> | [8, 2400]
forward conv7_2_mbox_loc             |  [8, 256, 5, 5] ->  | [8, 24, 5, 5]
forward conv7_2_mbox_loc_perm        |  [8, 24, 5, 5] ->   | [8, 5, 5, 24]
forward conv7_2_mbox_loc_flat        |  [8, 5, 5, 24] ->   | [8, 600]
forward conv8_2_mbox_loc             |  [8, 256, 3, 3] ->  |  [8, 16, 3, 3]
forward conv8_2_mbox_loc_perm        |  [8, 16, 3, 3] ->   | [8, 3, 3, 16]
forward conv8_2_mbox_loc_flat        |  [8, 3, 3, 16] ->   | [8, 144]
forward conv9_2_mbox_loc             |  [8, 256, 1, 1] ->  |  [8, 16, 1, 1]	
forward conv9_2_mbox_loc_perm        |  [8, 16, 1, 1] ->   | [8, 1, 1, 16]	
forward conv9_2_mbox_loc_flat        |  [8, 1, 1, 16] ->   | [8, 16]

23104 + 8664 + 2400 + 600 + 144 + 16 = 34928

However, we do not need to do anything since the size of mbox_loc = 8732 (per/class) x 4, is fixed regardless of the number of classes

#### mbox_priorbox is the concatenation of 
Layer Name  | Input Tensor | Output Tensor
---     |     ---      |    ---
forward conv4_3_norm_mbox_priorbox   |  [8, 512, 38, 38] ->  |[1, 2, 23104]
forward fc7_mbox_priorbox            |  [8, 1024, 19, 19] -> |[1, 2, 8664]
forward conv6_2_mbox_priorbox        |  [8, 512, 10, 10] ->  | [1, 2, 2400]
forward conv7_2_mbox_priorbox        |  [8, 256, 5, 5] ->    | [1, 2, 600]
forward conv8_2_mbox_priorbox        |  [8, 256, 3, 3] ->    | [1, 2, 144]
forward conv9_2_mbox_priorbox        |  [8, 256, 1, 1] ->    | [1, 2, 16]

23104 + 8664 + 2400 + 600 + 144 + 16 = 34828
However, we do not need to do anything since the size of mbox_loc = 8732 (per/class) x 4, is fixed regardless of the number of classes

***
## Parse prototxt to create a pytorch model only
### Requirements:
- Pytorch 

### Convert a Caffe model to a Pytorch model
given the Caffe model's prototxt file 
```
python3.7 demo_converted_pytorch_model.py --prototxt <path_to_caffe_prototxt_file> 
```
***
### Parse prototxt to create a pytorch model and load the pretrained caffe model
#### Requirements:
- Pytorch 
- Caffe 
 
```
from caffenet import *

def load_image(imgfile):
    import caffe
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models

imgfile = 'data/cat.jpg'
protofile = 'resnet50/deploy.prototxt'
weightfile = 'resnet50/resnet50.caffemodel'
image = load_image(imgfile)
pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)

```
