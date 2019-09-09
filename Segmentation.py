# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:01:42 2019

@author: Ashim
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


imageName = 'ComputerVisionBCU/basecolors.png'
lenaGS=cv2.imread(imageName,0)
plt.imshow(lenaGS,cmap='gray',vmin=0,vmax=255)
plt.show()

LL= 50
UL= 100

img2=(lenaGS < LL) & (lenaGS < UL) #it is a boolean matrix 
segLena = img2*255  #Python automatically cast true to 1, false to 0
plt.imshow(segLena,cmap='gray',vmin=0,vmax=255)
plt.show()


######################NOW COLOR PICTURES############

imageName = 'ComputerVisionBCU/mandrill1.jpg'
img=cv2.imread(imageName)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

print("RGB converted image:")
plt.imshow(img)
plt.show()

pixel = img[350:351,250:251,:] #to extract nose to get pixel value

#Here we got RGB = 244,97,87

plt.imshow(pixel)
plt.title('Sample Pixel')
plt.show()

r,g,b = cv2.split(img)

RL = 230
RU = 255

GL = 80
GU = 110

BL = 70
BU = 100

seg_R=((r > RL) & (r < RU)) 
seg_G=((g > GL) & (g < GU))
seg_B=((b > BL) & (b < BU))

seg_nose = (seg_R & seg_G & seg_B)*255

plt.imshow(seg_nose)
plt.show()

sample = img[350:420,200:300,:]  #this is for more sample of red color

r_hist,bins = np.histogram(sample[:,:,0].ravel(),255,[0,255])
g_hist,bins = np.histogram(sample[:,:,1].ravel(),255,[0,255])
b_hist,bins = np.histogram(sample[:,:,2].ravel(),255,[0,255])

plt.plot(bins[0:255],r_hist,'r',bins[0:255],g_hist,'g',bins[0:255],b_hist,'b')
plt.show()

stdR = np.std(sample[:,:,0])
meanR = np.mean(sample[:,:,0])
stdG = np.std(sample[:,:,1])
meanG = np.mean(sample[:,:,1])
stdB = np.std(sample[:,:,2])
meanB = np.mean(sample[:,:,2])


RL = meanR - stdR
RU = meanR + stdR

GL = meanG - stdG
GU = meanG + stdG

BL = meanB - stdB
BU = meanB + stdB

seg_R=((r > RL) & (r < RU)) 
seg_G=((g > GL) & (g < GU))
seg_B=((b > BL) & (b < BU))

seg_nose = (seg_R & seg_G & seg_B)*255

plt.imshow(seg_nose,cmap='gray',vmin=0,vmax=255)
plt.show()
