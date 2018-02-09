import cv2
import numpy as np
from matplotlib import pyplot as plt
from stitch import *
from harris_corner_detector import *

#Detects the corner points and writes to two separate files
img1 = cv2.imread('b1.jpg',0)
img2 = cv2.imread('b2.jpg',0)

rimg1 = cv2.resize(img1,(600,800))
rimg2 = cv2.resize(img2,(600,800))

blurred_img1 = gaussian_blur(rimg1,3,3)
blurred_img2 = gaussian_blur(rimg2,3,3)

bin1,cimg1,points1 = harris_corner_detector(blurred_img1,5,5,0.04,0.005)
bin2,cimg2,points2 = harris_corner_detector(blurred_img2,5,5,0.04,0.005)

plt.imsave('outputs/cimg1.jpg',cimg1)
plt.imsave('outputs/cimg2.jpg',cimg2)

f = open( 'outputs/corners1.txt', 'w' )
for i in range(len(points1)):
    f.write( repr(points1[i])+'\n' )
f.close()

f = open( 'outputs/corners2.txt', 'w' )
for i in range(len(points2)):
    f.write( repr(points2[i])+'\n' )
f.close()
