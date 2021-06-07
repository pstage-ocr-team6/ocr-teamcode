from skimage import transform
import numpy as np
import cv2
from pre_processing import *
from matplotlib import cm
from PIL import Image
class to_binary(object):
    """binary_image로 만듭니다.(약간의 전처리와 함께)
    """

    def __init__(self):
        pass
    def __call__(self, sample):
        gray = np.array(sample)
        # print(gray.shape)
        h,w=gray.shape[:2]
        orig_mean=gray.mean()
        black=np.zeros_like(gray)
        _max,_min=sliding_window1(gray)
        if orig_mean < 127:
            reszied_gray,show=remove_brightness(gray)
        else:
            if (_max-orig_mean) > 40 or (_min-orig_mean)<-40 :
                reszied_gray,show=remove_brightness(gray)
            else:
                reszied_gray,show=global_threshold1(gray)
                blurred = cv2.GaussianBlur(reszied_gray, (11,11), 1)
                reszied_gray_th=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15,2)
                masked_gray = np.where(show<127,reszied_gray_th,0)
                show=masked_gray+show
        im = Image.fromarray(show)
        return im