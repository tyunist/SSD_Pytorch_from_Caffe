"""Convert a Caffe model to Pytorch

Given a prototxt and a caffemodel, this code outputs a pytorch network model

Supported Caffe layers:
    'Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 'PReLU',
    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss',
    'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput'
    
"""
import os 
import pdb 
import pathlib 

from models.prototxt_to_pytorch import *
import argparse

# Default input prototxt path
HOME_DIR = str(pathlib.Path(__file__).parent.absolute().parent.absolute())
_PROTOTXT_PATH= '%s/SSD_caffe_model/ASU_model/deploy_6.prototxt'%HOME_DIR


def main():
    parser = argparse.ArgumentParser(
                       description='Convert caffe to pytorch.')
    parser.add_argument('--prototxt', type=str, default=_PROTOTXT_PATH,
                       help='Caffe prototxt path.')
    args = parser.parse_args()
    
    # Load network model
    print('========================================')
    print('Parsing Caffe model.')
    print('----------------------------------------')
    net = CaffeNet(args.prototxt)
    
    print('----------------------------------------')
    print('Print converted pytorch model.')
    print('----------------------------------------')
    print(net)
    
    # Try to run forward path 
    net.eval()
    img = torch.tensor(torch.ones([1, 3, 300, 300]))
    device = torch.device('cpu')
    img_tensor = img.to(device)
    output_dict     = net(img_tensor)
    print("Output mbox size:")
    print(output_dict['detection_out'].size())
    print('End.')
    print('========================================')

    
if __name__ == '__main__':
    main()

