# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:37:24 2019

@author: Ashim
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

upperLimit = 0.9
lowerLimit = 0.1
imageName = 'CoursePicture/left3.jpg'
origPicture=cv2.imread(imageName)
img=cv2.cvtColor(origPicture,cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
r,g,b = cv2.split(img)

red = cv2.equalizeHist(r)
green = cv2.equalizeHist(g)
blue = cv2.equalizeHist(b)

merged_img = cv2.merge((red, green,blue))

plt.imshow(merged_img)
plt.show()

histImage_r,bins = np.histogram(r.ravel(),255,[0,255])
hist_EQ_r, bins = np.histogram(red.ravel(),255,[0,255])

plt.plot(histImage_r)
plt.plot(hist_EQ_r)

img_YUV = cv2.cvtColor(origPicture, cv2.COLOR_BGR2YUV)

img_YUV[:,:,0]= cv2.equalizeHist(img_YUV[:,:,0])

img_output = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)

plt.imshow(img_output)
plt.show()
