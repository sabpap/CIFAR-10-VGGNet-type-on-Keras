#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:00 2017

Custom imshow function for easier use

@author: sabpap
import cv2

"""
from PIL import Image
import numpy as np

def imshow(img):
    
    np.rollaxis(img, 0,3)
    img = Image.fromarray(np.uint8(img))
    img.show()
    
    return