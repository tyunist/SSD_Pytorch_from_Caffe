from __future__ import absolute_import
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from collections import OrderedDict
from .prototxt import *
try:
    from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
except:
    from torch.nn.modules import CrossMapLRN2d as SpatialCrossMapLRNOld
from itertools import product as product
from .detection import Detection, MultiBoxLoss
from .utils import dict_has_key 
import pdb 

SUPPORTED_LAYERS = ['Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 
                    'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss', 
                    'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput', \
                     'Normalize', 'MyNormalize']


class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
            self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x

class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset
    def __repr__(self):
        return 'Crop(axis=%d, offset=%d)' % (self.axis, self.offset)

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, Variable(indices))
        return x

class Slice(nn.Module):
   def __init__(self, axis, slice_points):
       super(Slice, self).__init__()
       self.axis = axis
       self.slice_points = slice_points

   def __repr__(self):
        return 'Slice(axis=%d, slice_points=%s)' % (self.axis, self.slice_points)

   def forward(self, x):
       prev = 0
       outputs = []
       is_cuda = x.data.is_cuda
       if is_cuda: device_id = x.data.get_device()
       for idx, slice_point in enumerate(self.slice_points):
           rng = range(prev, slice_point)
           rng = torch.LongTensor(rng)
           if is_cuda: rng = rng.cuda(device_id)
           rng = Variable(rng)
           y = x.index_select(self.axis, rng)
           prev = slice_point
           outputs.append(y)
       return tuple(outputs)

class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)

class Permute(nn.Module):
    def __init__(self, order0, order1, order2, order3):
        super(Permute, self).__init__()
        self.order0 = order0
        self.order1 = order1
        self.order2 = order2
        self.order3 = order3

    def __repr__(self):
        return 'Permute(%d, %d, %d, %d)' % (self.order0, self.order1, self.order2, self.order3)

    def forward(self, x):
        x = x.permute(self.order0, self.order1, self.order2, self.order3).contiguous()
        return x

class Softmax(nn.Module):
    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Softmax(axis=%d)' % self.axis

    def forward(self, x):
        assert(self.axis == len(x.size())-1)
        orig_size = x.size()        
        dims = x.size(self.axis)
        x = F.softmax(x.view(-1, dims))
        x = x.view(*orig_size)
        return x

class SoftmaxWithLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(SoftmaxWithLoss, self).__init__()
    def __repr__(self):
        return 'SoftmaxWithLoss()'
    def forward(self, input, targets):
        targets = targets.long()
        return nn.CrossEntropyLoss.forward(self, input, targets)


class Normalize(nn.Module):
    ''' Normalize, a local response normalization
    '''
    def __init__(self,n_channels, scale=1.0):
        super(Normalize,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(1))
        self.weight.data *= 0.0
        self.weight.data += self.scale
        self.register_parameter('bias', None)

    def __repr__(self):
        return 'Normalize(channels=%d, scale=%f)' % (self.n_channels, self.scale)

    def forward(self, x):
        import pdb 
        pdb.set_trace()
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x
class MyNormalize(nn.Module):
    ''' MyNormalize, a local response normalization customized by ASU
    '''
    #TODO: please make sure if this is correct!
    def __init__(self,n_channels, scale=1.0, channel_shared=True, is_constant_scale=True):
        '''channel_shared: whether all channels share a scale value
           is_constant_scale: use a constant scale  
        '''
        super(MyNormalize,self).__init__()
        self.scale = scale
        self.is_constant_scale = is_constant_scale
        self.n_channels = n_channels 
        # Share weight between all channels or not
        if channel_shared:
            self.weight = nn.Parameter(torch.Tensor(1), requires_grad=not is_constant_scale)
        else:
            self.weight = nn.Parameter(torch.Tensor(self.n_channels), requires_grad=not is_constant_scale)
            self.weight.data *= 0.0
            self.weight.data += self.scale
            self.register_parameter('normalize', None)


    def __repr__(self):
        return 'MyNormalize(channels=%d, scale=%f)' % (self.n_channels, self.scale)

    def forward(self, x):
        if not self.is_constant_scale:
            norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
            x = x / (norm * self.weight.view(1,-1,1,1))
        else:
            #NOTE: don't do x *= self.scale as this inplace operation will cause backprop issue!
            x = x* self.scale
        return x


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Flatten(axis=%d)' % self.axis

    def forward(self, x):
        left_size = 1
        for i in range(self.axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1).contiguous()

# function interface, internal, do not use this one!!!
class LRNFunc(Function):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LRNFunc, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)


# use this one instead
class LRN(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def __repr__(self):
        return 'LRN(size=%d, alpha=%f, beta=%f, k=%d)' % (self.size, self.alpha, self.beta, self.k)

    def forward(self, input):
        return LRNFunc(self.size, self.alpha, self.beta, self.k)(input)

class Reshape(nn.Module):
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def __repr__(self):
        return 'Reshape(dims=%s)' % (self.dims)

    def forward(self, x):
        orig_dims = x.size()
        #assert(len(orig_dims) == len(self.dims))
        new_dims = [orig_dims[i] if self.dims[i] == 0 else self.dims[i] for i in range(len(self.dims))]
        
        return x.view(*new_dims).contiguous()

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
    def __repr__(self):
        return 'Accuracy()'
    def forward(self, output, label):
        max_vals, max_ids = output.data.max(1)
        n_correct = (max_ids.view(-1).float() == label.data).sum()
        batchsize = output.data.size(0)
        accuracy = float(n_correct)/batchsize
        print('accuracy = %f', accuracy)
        accuracy = output.data.new().resize_(1).fill_(accuracy)
        return Variable(accuracy)

class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """
    def __init__(self, min_size, max_size, aspects, clip, flip, step, offset, variances):
        super(PriorBox, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.aspects = aspects
        self.clip = clip
        self.flip = flip
        self.step = step
        self.offset = offset
        self.variances = variances

    def __repr__(self):
        return 'PriorBox(min_size=%f, max_size=%f, clip=%d, step=%d, offset=%f, variances=%s)' % (self.min_size, self.max_size, self.clip, self.step, self.offset, self.variances)
        
    def forward(self, feature, image):
        mean = []
        #assert(feature.size(2) == feature.size(3))
        #assert(image.size(2) == image.size(3))
        feature_height = feature.size(2)
        feature_width = feature.size(3)
        image_height = image.size(2)
        image_width = image.size(3)
        #for i, j in product(range(feature_height), repeat=2):
        for j in range(feature_height):
            for i in range(feature_width):
                # unit center x,y
                cx = (i + self.offset) * self.step / image_width
                cy = (j + self.offset) * self.step / image_height
                mw = float(self.min_size)/image_width
                mh = float(self.min_size)/image_height
                mean += [cx-mw/2.0, cy-mh/2.0, cx+mw/2.0, cy+mh/2.0]

                if self.max_size > self.min_size:
                    ww = math.sqrt(mw * float(self.max_size)/image_width)
                    hh = math.sqrt(mh * float(self.max_size)/image_height)
                    mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]
                    for aspect in self.aspects:
                        ww = mw * math.sqrt(aspect)
                        hh = mh / math.sqrt(aspect)
                        mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]
                        if self.flip:
                            ww = mw / math.sqrt(aspect)
                            hh = mh * math.sqrt(aspect)
                            mean += [cx-ww/2.0, cy-hh/2.0, cx+ww/2.0, cy+hh/2.0]

        # back to torch land
        output1 = torch.Tensor(mean).view(-1, 4)
        output2 = torch.FloatTensor(self.variances).view(1,4).expand_as(output1)
        output2 = output2.to(output1.device)
        if self.clip:
            output1.clamp_(max=1, min=0)
        output1 = output1.view(1,1,-1)
        output2 = output2.contiguous().view(1,1,-1)
        output = torch.cat([output1, output2], 1)
        if feature.data.is_cuda:
            device_id = feature.data.get_device()
            return Variable(output.cuda(device_id))
        else:
            return Variable(output)

