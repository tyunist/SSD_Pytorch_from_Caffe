# -*- coding:utf-8 -*- 
"""
This is mainly adopted from: 
https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/layers/functions/detection.py
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import pdb 

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """
    Convert prior_boxes to (cx, cy, w, h)
    Input:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    Return:
        boxes: N x (cx, cy, w, h)
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    #best_prior_idx.squeeze_(1)
    #best_prior_overlap.squeeze_(1)
    #best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j # index of the object
        best_truth_overlap[best_prior_idx[j]] = best_prior_overlap[j]
  
    # Get Groundtruth box coordinates that matched with the priorboxes
    # best_truth_idx would be sth like 0,0,0..0, 1,...1, 2,...,2, ....
    # truths size is: num_objects x 4. The following will do broadcast 
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    
    # Assign lables to num_priors priorboxes
    # labels size is: num_objects x 4. The following will do broadcast 
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    # Label as background. Assumption: objects' labels start from 1
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions: num_priors x 4 of format (x1, y1, x2, y2) can be negative -> positive 
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    # Additional trick is to use x_max to prevent overflow while preserving as many accurate leading values 
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def clip_boxes(boxes):
    boxes = torch.clamp(boxes, min = 0.0, max = 1.0)
    return boxes

class Detection(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, keep_top_k):
        super(Detection, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.keep_top_k = keep_top_k
        self.variance = [0.1, 0.2]

    def forward(self, loc, conf, prior, targets=None):
        """
        Args:
            loc: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors*num_classes]
            prior: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,2, num_priors*4]
        Return:
            output = torch.zeros(B, 7)
            where the last dim: [batch_i, cls idx, score, x1, y1, x2, y2] 
            with the range of x1,x2,y1,y2 is not so sure yet! 
        """

        num = loc.size(0)
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data

        num_classes = self.num_classes
        num_priors  = int(prior_data.size(2)/4)
        
        # Convert priors to num_priors x 4 
        prior_data  = prior_data[0][0].view(num_priors, 4)
        # Convert priors to (cx, cy, w, h)
        prior_data  = center_size(prior_data)

        # Convert loc data to batch x num_priors x 4  
        loc_data    = loc_data.view(num, num_priors, 4)
        
        # Output
        #output = torch.zeros(num, self.num_classes, self.top_k, 5)
        output  = []

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.view(num_priors, self.num_classes).t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)
        
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            #decoded_boxes: (tensor) xmin, ymin, xmax, ymax form of boxes.

            # for each class, perform nms
            conf_scores   = conf_preds[i].clone()
            
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.size(0) == 0:
                    continue 
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes  = decoded_boxes[l_mask].view(-1,4)
                
                # idx of highest scoring and non-overlapping boxes per class 
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # Take only self.keep_top_k best 
                count = min(count, self.keep_top_k) # since ids refer to sorted boxes
                
                # Add batch index and class label to it
                extra_info = torch.FloatTensor([i, cl]).view(1,2).expand(count,2)
                extra_info = extra_info.to(boxes.device)
                boxes = torch.cat((extra_info, scores[ids[:count]].unsqueeze(1),
                    boxes[ids[:count]]), 1)

                #output[i, cl, :count] = \
                        #torch.cat((scores[ids[:count]].unsqueeze(1), \
                                   #boxes[ids[:count]]),1)
                output.append(boxes)

        output = torch.cat(output, 0) # B x 7
        print("------------------------------------------")
        print(">> Detection_output shape:", output.shape)
        print("------------------------------------------")
        # Debug: output targets 
        #pdb.set_trace()
        #targets: B x 6
        #where the last dim: [batch_i, cls idx, score, x1, y1, x2, y2] 
        #count  = targets.size(0)
        #fake_scores = torch.FloatTensor([1.]).view(1,1).expand(count,1)
        #fake_scores = fake_scores.to(targets.device)
        #output = torch.cat((targets[:,:2], fake_scores, targets[:, 2:]), 1) 
        return output


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + ¦Áloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by ¦Áwhich is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, use_gpu=True, local_weight=1.0):
        '''
        Inputs:
            local_weight: weight for loss_l
        '''
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        
        self.loss_l_weight = local_weight

    def forward(self, loc_data, conf_data, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            targets(tensor): Ground truth boxes and labels for a batch,
                shape: [sum_batches(num_objs),6] 
                0th index is the batch index
                1st index is the label
                2:6 index is the bbox (x1y1x2y2) normalized to [0,1]
                ).
            priors: num_priors x 4 tensor i.e. [[x1, y1, x2, y2], ...] where x1, x2, y1, y2 are normalized to [0,1] even though some x1,y1 can be < 0 
        """
        #loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        num_priors = int((loc_data.size(1)/4))
        num_classes = self.num_classes
        loc_data = loc_data.view(num, num_priors, 4)
        conf_data = conf_data.view(num, num_priors, num_classes)
        # Convert priors to num_priors x 4 shape
        priors = priors[0][0].view(num_priors, 4)
        targets = targets.view(-1, 6)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.zeros(num, num_priors).long()
        for idx in range(num):
            sub_mask = (targets[:,0] == idx)
            if sub_mask.data.float().sum() == 0:
                continue
            sub_targets = targets[sub_mask.view(-1,1).expand_as(targets)].view(-1,6)
            truths = sub_targets[:, 2:6].data
            labels = sub_targets[:, 1].data
            defaults = priors.data
            defaults = center_size(defaults)
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
    

        # Compute loc loss only on positive (> 1) boxes which are associated with objects
        pos = conf_t > 0
        num_pos = pos.sum(dim=1,keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        
        # Predicted boxes in a an encoded format (def encode() function)
        loc_p = loc_data[pos_idx].view(-1, 4)
        
        # Gt boxes in an encoded format
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 1st part is the denominator of the log(C), 2nd part is the nominator of log(C)
        loss_c = log_sum_exp(batch_conf).view(-1,1) - batch_conf.gather(1, conf_t.view(-1, 1))

        #Ty add the following line
        loss_c = loss_c.view(num, num_priors)

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        
        #Somehow doing one time sort might not give results as expected -> sort twice.
        #i.e. m = tensor([-1,  2,  3, -4,  6,  6,  7, -1,  3])
        # _, m_idx = m.sort(0, descending=True) 
        # Return: tensor([6, 4, 5, 2, 8, 1, 0, 7, 3]) => very weird! 
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = Variable(torch.clamp(self.negpos_ratio*num_pos.data.float(), max=pos.size(1)-1).long())
        neg = idx_rank < num_neg.expand_as(idx_rank)
        

        # Confidence Loss Including Positive and Negative Examples
        pos_mask = pos.unsqueeze(2).expand_as(conf_data)
        neg_mask = neg.unsqueeze(2).expand_as(conf_data)
        
        # Select boxes that are predicted to contain objects
        conf_p = conf_data[(pos_mask+neg_mask).gt(0)].view(-1, self.num_classes)
        # Corresponding gt boxes 
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + ¦Áloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        total_loss = loss_l*self.loss_l_weight + loss_c
        print(">>|total Loss %.4f |Loss_l %.4f|Loss_c %.4f"%(total_loss, loss_l, loss_c))
        return {'total_loss':total_loss,\
                'loss_l':loss_l,\
                'loss_c':loss_c 
                }

