"""Convert a Caffe model to Pytorch

Given a prototxt and a caffemodel, this code outputs a pth model.
You can reconstruct the network by the prototxt and the pth model.

Supported Caffe layers:
    'Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 'PReLU',
    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss',
    'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput'
    
***Notice: This code requires python2***
"""
import os 
import pdb 
import pathlib 

from models.caffenet import *
import argparse
import caffe.proto.caffe_pb2 as caffe_pb2
from torch.utils.data import DataLoader 
from utils.datasets import ImageFolder, ListDataset


# Default input prototxt path
HOME_DIR = str(pathlib.Path(__file__).parent.absolute().parent.absolute())
_PROTOTXT_PATH= '%s/SSD_caffe_model/ASU_model/deploy_6.prototxt'%HOME_DIR

# Default input caffemodel path
_CAFFEMODEL_PATH = '%s/SSD_caffe_model/ASU_model/SSD300_6_iter_120000.caffemodel'%HOME_DIR

# Default output pytorch weights path
_PTHMODEL_PATH   = '%s/SSD_pytorch_model/SSD300_6_iter_120000.pth'%HOME_DIR

torch_folder_dataloader = DataLoader(
            ImageFolder('data/samples', img_size=300),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

torch_list_dataloader = DataLoader(
            ListDataset('data/samples/listfile.txt', img_size=300, normalized_labels=False),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)


def parse_caffemodel(caffemodel):
    """Parse a caffemodel
    
    Inputs:
      string of caffemodel path
    Returns:
      Parsed model
    
    This function should be in prototxt.py, but this function
    requires caffe which requires python2. So move it here.
    """
    model = caffe_pb2.NetParameter()
    print('Loading caffemodel: ' + caffemodel)
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model


def main():
    parser = argparse.ArgumentParser(
                       description='Convert caffe to pytorch.')
    parser.add_argument('--prototxt', type=str, default=_PROTOTXT_PATH,
                       help='Caffe prototxt path.')
    parser.add_argument('--caffemodel', type=str, default=_CAFFEMODEL_PATH,
                       help='Caffe caffemodel path.')
    parser.add_argument('--pthmodel', type=str, default=_PTHMODEL_PATH,
                       help='Output pytorch model path.')
    args = parser.parse_args()
    
    # Load network model
    print('========================================')
    print('Load torch model.')
    print('----------------------------------------')
    net = CaffeNet(args.prototxt)
    
    # Load weights 
    net.load_state_dict(torch.load(args.pthmodel))
    print('----------------------------------------')
    print('Print converted pytorch model.')
    print('----------------------------------------')
    print(net)
    
    # Run a forward path 
    imgfile = 'data/cat.jpg'
    net.eval()
    for batch_i, (img_paths, input_imgs, targets) in enumerate(torch_list_dataloader):
        input_imgs = Variable(input_imgs)
        blobs = net(input_imgs, targets)

    
    print('End.')
    print('========================================')

    
if __name__ == '__main__':
    main()

