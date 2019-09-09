
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(img, length):
    newImage = img * 1
    start = length // 2
    for i in range(start, (len(img) - start)): #row of the point
        for j in range(start, (len(img[0]) - start)): #column of the point
            w, h = length, length;
            Matrix = [[0 for x in range(w)] for y in range(h)]         
            x = 0
            for k in range (i-start, i+start+1):
                y = 0
                for l in range(j-start, j+start+1):
                    Matrix[x][y] = img[k][l]
                    y = y + 1
                x = x + 1
            value = np.median(Matrix)
            newImage[i][j] = value
    return newImage


imageName = 'Pic1.jpg'
pic1=cv2.imread(imageName,0)

imageName = 'Pic2.jpg'
pic2=cv2.imread(imageName,0)






newPic1 = convolution(pic1, 3)
plt.imshow(newPic1,cmap='gray',vmin=0,vmax=255)
plt.show()


newPic2 = convolution(pic2,3)
plt.imshow(newPic2,cmap='gray',vmin=0,vmax=255)
plt.show()
