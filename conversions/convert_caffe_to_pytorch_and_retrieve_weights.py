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

# Default input prototxt path
HOME_DIR = str(pathlib.Path(__file__).parent.absolute().parent.absolute())
_PROTOTXT_PATH= '%s/SSD_caffe_model/ASU_model/deploy_6.prototxt'%HOME_DIR

# Default input caffemodel path
_CAFFEMODEL_PATH = '%s/SSD_caffe_model/ASU_model/SSD300_6_iter_120000.caffemodel'%HOME_DIR
# Default output pytorch weights path
_PTHMODEL_PATH   = '%s/SSD_pytorch_model/SSD300_6_iter_120000.pth'%HOME_DIR

def caffe_load_image(imgfile):
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
    print('Parsing Caffe model.')
    print('----------------------------------------')
    net = CaffeNet(args.prototxt)
    
    print('----------------------------------------')
    print('Print converted pytorch model.')
    print('----------------------------------------')
    print(net)
    
    # Load network weights
    print('----------------------------------------')
    print('Loading weights.')
    print('----------------------------------------')
    net.load_weights(parse_caffemodel(args.caffemodel))
    
    # Save model structure as OrderedDict
    print('----------------------------------------')
    print('Saving pytorch weights to %s'%args.pthmodel)
    print('----------------------------------------')
    torch.save(net.state_dict(), args.pthmodel)
 
    print('End.')
    print('========================================')

    
if __name__ == '__main__':
    main()

