import torchvision.transforms as transforms
import numpy as np
import cv2
from pre_processing import *
from PIL import Image
from exp.nb_SparseImageWarp import sparse_image_warp
import random
import torch


class to_binary(object):
    """ Gray scale image to binary image.
    """
    def __call__(self, sample):
        gray = np.array(sample)
        orig_mean = gray.mean()
        _max,_min = sliding_window1(gray)
        if orig_mean < 127:
            reszied_gray, show = remove_brightness(gray)
        else:
            if (_max - orig_mean) > 40 or (_min - orig_mean) < -40 :
                reszied_gray, show = remove_brightness(gray)
            else:
                reszied_gray,show = global_threshold1(gray)
                blurred = cv2.GaussianBlur(reszied_gray, (11, 11), 1)
                reszied_gray_th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15, 2)
                masked_gray = np.where(show < 127, reszied_gray_th, 0)
                show = masked_gray + show
        im = Image.fromarray(show)
        return im


class cutout(object):
    """Cutout

    Args:
        mask_size (int): Size of boxes to cut out of each image.
        p (float): Cutout probability. (0~1)
        cutout_inside (bool): If true, generate maskes inside image. If false, maskes can be generated outside of image.
        max_boxes (int): Number of boxes to cut out of each image.
    """
    def __init__(self, mask_size, p, cutout_inside, max_boxes):
        self.mask_size = mask_size
        self.p = p
        self.cutout_inside = cutout_inside
        self.max_boxes = max_boxes
        
    def __call__(self,image):
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0
        image = np.array(image).copy()
        for _ in range(self.max_boxes):
            if np.random.random() > self.p:
                image = Image.fromarray(image)
                return image

            h, w = image.shape[:2]

            if self.cutout_inside:
                cxmin, cxmax = mask_size_half, w + offset - mask_size_half
                cymin, cymax = mask_size_half, h + offset - mask_size_half
            else:
                cxmin, cxmax = 0, w + offset
                cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + self.mask_size
            ymax = ymin + self.mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            image[ymin:ymax, xmin:xmax] = self.mask_size
        image = Image.fromarray(image)
        return image


class specAugment(object):
    """https://github.com/zcaceres/spec_augment
        Args:
            row_num_masks (int) : Number of row_maskes to draw image of each image.
            col_num_masks (int) : Number of column_maskes to draw image of each image.
            replace_with_zero (bool) : If true, masked with zero. Defaults to True.
    """
    def __init__(self, row_num_masks, col_num_masks, replace_with_zero=True):
        self.row_num_masks = row_num_masks
        self.col_num_masks = col_num_masks
        self.replace_with_zero = replace_with_zero

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            pass
        else:
            tf=transforms.ToTensor()
            image=tf(image)
        combined = col_mask(row_mask(image, num_masks=2, replace_with_zero=True), num_masks=1, replace_with_zero=True)
        combined = transforms.ToPILImage()(combined).convert("L")
        return combined
    

def row_mask(spec, F=10, num_masks=1, replace_with_zero=False):
    """ Row masking is applied so that f consecutive mel frequency channels [f0, f0 + f) are masked, 
        where f is first chosen from a uniform distribution from 0 to the frequency mask parameter F, 
        and f0 is chosen from 0, ν − f). ν is the number of mel frequency channels.

        Args:
            spec (image, tensor type) : Image
            F (int) : Max length of line. Defaults to 10.
            num_masks (int) : Number of maskes to draw image of each image. Defaults to 1.
            replace_with_zero (bool) : If true, masked with zero. Defaults to False.
        Return:
            cloned (image,tensor) : image with line
    """
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    num_masks=random.randrange(0, 2)
    for i in range(0, num_masks):
        f = random.randrange(1, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f) 
        if (replace_with_zero): cloned[0][f_zero:mask_end] = 0
        else: cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def col_mask(spec, T=10, num_masks=1, replace_with_zero=False):
    """ Column masking is applied so that t consecutive time steps [t0, t0 + t) are masked, 
        where t is first chosen from a uniform distribution from 0 to the time mask parameter T, and t0 is chosen from [0, τ − t). 
        We introduce an upper bound on the time mask so that a time mask cannot be wider than p times the number of time steps.

        Args:
            spec (image, tensor type) : Image
            T (int) : Max length of line. Defaults to 10.
            num_masks (int) : Number of maskes to draw image of each image. Defaults to 1.
            replace_with_zero (bool) : If true, masked with zero. Defaults to False.
        Return:
            cloned (image,tensor) : image with line
    """
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    num_masks=random.randrange(0, 2)
    for i in range(1, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero): cloned[0][:, t_zero:mask_end] = 0
        else: cloned[0][:, t_zero:mask_end] = cloned.mean()
    
    return cloned