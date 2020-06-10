import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random
from utils.draw_bbox import draw_bbox  as bbox_drawer

EXPAND = 0
SHRINK = 1
KEEP_WIDTH = 2
KEEP_HEIGHT = 3
KEEP_AREA = 4


def __normalize_format(bbs):
    if not isinstance(bbs, np.ndarray) or bbs.size == 0:
        bbs = np.array(bbs, copy=True)
        if bbs.size == 0:
            bbs = np.empty((0,5), np.float32)
    return np.atleast_2d(bbs).astype(np.float32, copy=False)


def __width(bbs):
    return bbs[:,2]


def __height(bbs):
    return bbs[:,3]


def __aspect_ratio(bbs):
    return bbs[:,2] / bbs[:,3]


def __set_width(h, ar):
    return h*ar, h


def __set_height(w, ar):
    return w, w/ar


def __set_area(w,h,ar):
    area = w*h
    nw = np.sqrt(area * ar)
    nh = area / nw
    return nw, nh


def set_aspect_ratio(bbs, ar=1.0, type=KEEP_AREA):
    """
    Set apect ration without moving bb center
    Input:
        bbs     - Bounding boxes
        ar      - Target aspect ratio (width/height)
        type    - One of bbx.EXPAND, bbx.SHRINK, bbx.KEEP_WIDTH, bbx.KEEP_HEIGHT, bbx.KEEP_AREA
    Output:
        bbs width altered aspect ration set to ar
    """

    bbs = __normalize_format(bbs)
    w = __width(bbs)
    h = __height(bbs)

    if type is KEEP_AREA:
        nw, nh = __set_area(w, h, ar)
    elif type is KEEP_WIDTH:
        nw, nh = __set_height(w, ar)
    elif type is KEEP_HEIGHT:
        nw, nh = __set_width(h, ar)
    elif type is EXPAND:
        mask = w/h > ar
        nw = np.empty_like(w)
        nh = np.empty_like(h)
        nw[ mask], nh[ mask] = __set_height(w[mask], ar)
        nw[~mask], nh[~mask] = __set_width(h[~mask], ar)
    elif type is SHRINK:
        mask = w/h > ar
        nw = np.empty_like(w)
        nh = np.empty_like(h)
        nw[ mask], nh[ mask] = __set_width(h[mask], ar)
        nw[~mask], nh[~mask] = __set_height(w[~mask], ar)
    else:
        raise NotImplementedError

    sx, sy = (nw-w)/2, (nh-h)/2
    bbs[:,0] -= sx
    bbs[:,1] -= sy
    bbs[:,2] = nw
    bbs[:,3] = nh

    return bbs


def resize(bbs, ratio=1):
    """
    Resize all bounding boxes in bbs by ratio, keeping their center.
    Input:
        bbs - bounding boxes
        ratio - scalar, tuple or np.ndarray with resize ratio
    Output:
        np.ndarray with resized bounding boxes
    Raises:
        ValueError when wrong ratio is given

    There are different scenarios for various ratio inputs. When the ratio is
    scalar, all bounding boxes are resized by the same factor. In case of two
    scalars (tuple or array), all bbs are resized with different factor for width
    and height. Vector with length corresponding to the number of bbs, each bounding
    box is resized by its respective factor (same for width and height). The last
    case is when the ratio is matrix with two columns and rows corresponding to the
    number of bbs. Then each bb is resized by its respective factor different for
    width and height. In other cases, ValueError is raised.

    Example:
        bbs = [ [0,0,10,10], [10,10,10,10] ]
        bbx.resize(bbs, 2)      # [ [-5,-5,20,20],[5,5,20,20] ] - Just double the size of both
        bbx.resize(bbs, [1,2])  # [ [0,-5,10,20], [10,5,10,20] ] - Double the height
        bbx.resize(bbs, [[1,1],[2,2]])  # [ [0,0,10,10], [5,5,20,20] ] - Reisze only the second

    """
    bbs = __normalize_format(bbs)
    n = bbs.shape[0]

    r = np.array(ratio)
    if r.size == 1:
        rx = ry = r
    elif r.size == 2:
        rx,ry = r
    elif r.size == n:
        rx = ry = r.flatten()[:,None]
    elif r.ndim == 2 and r.shape == (n,2):
        rx = r[:,0]
        ry = r[:,1]
    else:
        raise ValueError("Wrong resize ratio")

    w = __width(bbs)
    h = __height(bbs)
    nw, nh = w*rx,  h*ry
    sx, sy = (nw-w)/2, (nh-h)/2

    bbs[:,0] -= sx
    bbs[:,1] -= sy
    bbs[:,2] = nw
    bbs[:,3] = nh

    return bbs


def scale(bbs, s=1):
    """
    Scale all bbs by the given factor
    """
    bbs = __normalize_format(bbs)
    bbs[:,:4] * s
    return bbs


def move(bbs, shift=0):
    bbs = __normalize_format(bbs)
    bbs[:,:2] += shift
    return bbs


def center(bbs):
    bbs = __normalize_format(bbs)
    return bbs[:,:2] + 0.5*bbs[:,2:4]

def scaleBboxwithImg(img, bbox, output_h, output_w, is_center_crop=False):
    '''Scale bounding box while scaling the image'''
    bbox = np.array(bbox)
    if len(bbox.shape) < 2:
        bbox = bbox[np.newaxis, :]

    input_h, input_w = img.shape[:2]
    scale_h, scale_w = output_h*1.0/input_h, output_w*1.0/input_w

    if max(scale_h, scale_w) >= 1:
        scale = max(scale_h, scale_w)
    else:
        scale = min(scale_h, scale_w)

    if scale >= 1.0:
        scaled_img  = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    else:
        scaled_img  = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    
    new_img = np.zeros([output_h, output_w, img.shape[-1]])

    # There might be a crop due to the difference betwen output ratio and input ratio
    if is_center_crop:
        start_x = int(input_w*scale - output_w)//2 
        start_y = int(input_h*scale - output_h)//2 
    else:
        start_x = 0 
        start_y = 0 

    new_img = scaled_img[-start_y:output_h-start_y, -start_x:output_w-start_x]   
    
    if len(bbox[0]) == 0:
        return new_img, [[]]
    
    # Scale the bbox
    bbox = np.array(bbox)*scale 
    bbox = bbox.astype(np.int)

    # If center-crop, there is some  shifting 
    if is_center_crop:
        delta = np.array([start_x, start_y, 0, 0])[np.newaxis, :]      
        bbox += delta
    
    return new_img, bbox 

def normalizeBbox(bbox, h, w):
    '''bbox: N x [x, y, w, h]'''
    bbox = np.array(bbox)
    divide_array = np.array([w, h, w, h], dtype=np.float32).reshape(bbox.shape)
    bbox = bbox/divide_array
    return bbox

class PltDrawBboxes(object):
    def __init__(self, classes, input_h, input_w):
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        random.seed(2)
        self.bbox_colors = random.sample(colors, len(classes))
        
        self.classes     = classes # class names
        self.input_h = input_h
        self.input_w = input_w

    def drawBboxes(self, img, detections, savefig_name=None, is_add_label=True):
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                # TODO: if we want each box has an unique color regardless of its class, uncomment first 2 lines
                # and comment the 3rd line
                #bbox_colors = np.random.sample(colors, n_cls_preds)
                #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color = self.bbox_colors[int(cls_pred)]

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                if is_add_label:
                    plt.text(
                        min(max(x1, 10), int(self.input_w)-100),
                        min(max(10, y1-30), int(self.input_h)-20),
                        s=self.classes[int(cls_pred)] + " [%.0f %%]"%(cls_conf*100),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                        fontsize=6,
                    )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(savefig_name, box_inches="tight", pad_inches=0.0)
        #plt.pause(2)
        plt.close()

class CvDrawBboxes(object):
    def __init__(self, classes, input_h, input_w):
        color_names = bbox_drawer._COLOR_NAME_TO_RGB.keys()
        random.seed(3)
        self.bbox_colors = random.sample(color_names, len(classes))
        
        self.classes     = classes # class names
        self.input_h = input_h
        self.input_w = input_w

    def drawBboxes(self, img, detections, savefig_name):
        if detections is not None:
            # Rescale boxes to original image
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))
                label =self.classes[int(cls_pred)] + " [%.0f %%]"%(cls_conf*100)
                color = self.bbox_colors[int(cls_pred)]
                bbox_drawer.add(img, x1, y1, x2, y2, label, color) 

        cv.imwrite(savefig_name, img)
