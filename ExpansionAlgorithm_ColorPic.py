# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:05:54 2019

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


r_float = r.astype(np.float)
g_float = g.astype(np.float)
b_float = b.astype(np.float)



histImage_r,bins = np.histogram(r.ravel(),255,[0,255])
plt.plot(histImage_r)
plt.show()

r_histCum = np.cumsum(histImage_r)
upperY = find_nearest(r_histCum, upperLimit*(np.max(r_histCum)))
lowerY = find_nearest(r_histCum, lowerLimit*(np.max(r_histCum)))

upperValue = (np.where(r_histCum == upperY))[0]
lowerValue = (np.where(r_histCum == lowerY))[0]

exp_r = ((r_float - lowerValue)/(upperValue - lowerValue)) * 255

r_newHist,bins = np.histogram(exp_r.ravel(),255,[0,255])
plt.plot(r_newHist)
plt.show()
#plt.plot(histImage_r)
#plt.plot(exp_r)



histImage_g,bins = np.histogram(g.ravel(),255,[0,255])
plt.plot(histImage_g)
plt.show()
g_histCum = np.cumsum(histImage_g)
upperY = find_nearest(g_histCum, upperLimit*(np.max(g_histCum)))
lowerY = find_nearest(g_histCum, lowerLimit*(np.max(g_histCum)))

upperValue = (np.where(g_histCum == upperY))[0]
lowerValue = (np.where(g_histCum == lowerY))[0]


exp_g = ((g_float - lowerValue)/(upperValue - lowerValue)) *255

g_newHist,bins = np.histogram(exp_g.ravel(),255,[0,255])
plt.plot(g_newHist)
plt.show()


histImage_b,bins = np.histogram(b.ravel(),255,[0,255])
plt.plot(histImage_b)
plt.show()
b_histCum = np.cumsum(histImage_b)
upperY = find_nearest(b_histCum, upperLimit*(np.max(b_histCum)))
lowerY = find_nearest(b_histCum, lowerLimit*(np.max(b_histCum)))

upperValue = (np.where(b_histCum == upperY))[0]
lowerValue = (np.where(b_histCum == lowerY))[0]


exp_b = ((b_float - lowerValue)/(upperValue - lowerValue)) * 255

b_newHist,bins = np.histogram(exp_b.ravel(),255,[0,255])
plt.plot(b_newHist)
plt.show()
#plt.plot(r_histCum)


merged_img = cv2.merge((exp_r, exp_g,exp_b))
plt.imshow(merged_img/255)
plt.show()


