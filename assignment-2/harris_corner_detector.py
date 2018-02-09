#Importing modules
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""Performs harris corner detection 
img: Image
window_size: Window size for corner detection
sobel_ksize: Sobel kernel size 
k: Parameter in Harris Corner Detector
t: Threshold for response 
""" 
def harris_corner_detector(img,window_size,sobel_ksize,k,t):
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,sobel_ksize)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,sobel_ksize)
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = np.multiply(Ix,Iy)
    
    """The sum of products of matrix values in the window multiplied by Gaussian weights is
    simply applying a Gaussian smoothening over the window """
    Sxx = cv2.GaussianBlur(Ixx,(window_size,window_size),0)
    Syy = cv2.GaussianBlur(Iyy,(window_size,window_size),0)
    Sxy = cv2.GaussianBlur(Ixy,(window_size,window_size),0)
    
    det = (Sxx * Syy)-(Sxy**2)
    trace = Sxx+Syy
    r = det - k*(trace**2)
    
    res_img = img.copy()
    res_img = cv2.cvtColor(res_img,cv2.COLOR_GRAY2RGB)
    
    corner_points = []
    rows,cols = r.shape
    harris_binary = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if(r[i,j]>t*r.max()):
                harris_binary[i,j]=1
                corner_points.append([i,j,r[i,j]])
                res_img.itemset((i,j,0),255)
                res_img.itemset((i,j,1),0)
                res_img.itemset((i,j,2),0)

    return harris_binary,res_img,corner_points