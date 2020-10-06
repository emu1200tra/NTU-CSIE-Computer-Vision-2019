import cv2
import numpy as np
import math
'''
Part 1
'''
image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
print('shape:', image.shape)

#a. upside-down
updown = image[::-1,:]
cv2.imwrite("upsideDown.jpg", updown)

#b. right-side-left 
rightLeft = image[:,::-1]
cv2.imwrite("rightSideLeft.jpg", rightLeft)

#c. diagonally mirrored
w,h = image.shape
diagonal = np.zeros(image.shape)
for i in range(w):
    for j in range(h):
        diagonal[i,j] = image[j,i]

cv2.imwrite("diagonal.jpg", diagonal)

'''
Part 2
'''
#a. rotate
#use software to complete

#b. shrink
#use software to complete

#c. binarize 
index = np.where(image>=128)
image[index] = 255
index = np.where(image<128)
image[index] = 0
cv2.imwrite("binarized.jpg", image)
