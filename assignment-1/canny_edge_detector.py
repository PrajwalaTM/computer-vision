#Importing modules
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""Performs gaussian blur to denoise image before edge detection
img : Image 
gaussian_ksize: Kernel size 
"""
def gaussian_blur(img,gaussian_ksize):
    blurred_img = cv2.GaussianBlur(img,(gaussian_ksize,gaussian_ksize),0)
    return blurred_img

"""Rounds off the gradient direction to one of the four angles
angle_in_rad: Input angle in radian
"""
def angle_approx(angle_in_rad):
    angle_in_deg = np.rad2deg(angle_in_rad) % 180
    if (0 <= angle_in_deg < 22.5) or (157.5 <= angle_in_deg < 180):
        angle_in_deg = 0
    elif (22.5 <= angle_in_deg < 67.5):
        angle_in_deg = 45
    elif (67.5 <= angle_in_deg < 112.5):
        angle_in_deg = 90
    elif (112.5 <= angle_in_deg < 157.5):
        angle_in_deg = 135
    return angle_in_deg

"""Computes gradient magnitude and direction for Non-max suppression
img: Image 
sobel_ksize: Sobel kernel size for gradient computation
"""
def intensity_gradient(img,sobel_ksize):
    grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_ksize)
    grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_ksize)
    grad_mag = np.hypot(grad_x, grad_y)
    grad_dir = np.arctan2(grad_y,grad_x)
    
    rows,cols = grad_dir.shape
    res_img = np.zeros((rows,cols))
    round_grad_dir = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            round_grad_dir[i,j] = angle_approx(grad_dir[i,j])
    return (grad_x,grad_y,grad_mag,round_grad_dir)

"""Performs Non-max suppression 
grad_mag: Magnitude of gradient
grad_dir: Rounded off gradient direction in degrees
"""
def non_max_suppression(grad_mag,grad_dir):
    rows,cols = grad_dir.shape
    res_img = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            if(grad_dir[i,j]==0):
                if (j!=0 and grad_mag[i,j]>=grad_mag[i,j-1]) and (j!=cols-1 and grad_mag[i,j]>=grad_mag[i,j+1]):
                    res_img[i,j]=grad_mag[i,j]
            elif (grad_dir[i,j]==90):
                if (i!=0 and grad_mag[i,j]>=grad_mag[i-1,j]) and (i!=rows-1 and grad_mag[i,j]>=grad_mag[i+1,j]):
                    res_img[i,j]=grad_mag[i,j]
            elif(grad_dir[i,j]==45):
                if (i!=0 and j!=cols-1 and grad_mag[i,j]>=grad_mag[i-1,j+1]) and (i!=rows-1 and j!=0 and grad_mag[i,j]>=grad_mag[i+1,j-1]):
                    res_img[i,j]=grad_mag[i,j]
            elif(grad_dir[i,j]==135):
                if (i!=0 and j!=0 and grad_mag[i,j] >= grad_mag[i-1,j-1]) and (i!=rows-1 and j!=cols-1 and grad_mag[i,j] >= grad_mag[i+1,j+1]):
                        res_img[i,j] = grad_mag[i,j]
    return res_img

"""Performs hysterisis thresholding
grad_mag: Output of non-max-suppression containing only local maxima
min_val: Lower value of threshold
max_val: Upper value of threshold
"""
def hysterisis_thresholding(grad_mag, min_val, max_val):
    strong_i,strong_j = np.where(grad_mag>max_val)
    weak_i,weak_j = np.where((grad_mag>=min_val) & (grad_mag<=max_val))
    zero_i,zero_j = np.where(grad_mag<min_val)
    
    grad_mag[strong_i,strong_j]=np.int32(255)
    grad_mag[weak_i,weak_j]=np.int32(128)
    grad_mag[zero_i,zero_j]=np.int32(0)
    
    rows,cols = grad_mag.shape
    for i in range(rows):
        for j in range(cols):
            if(grad_mag[i,j]==128):
                if((i!=rows-1 and grad_mag[i+1,j]==255) or (i!=0 and grad_mag[i-1,j]==255)
                   or (j!=cols-1 and grad_mag[i,j+1]==255) or (j!=0 and grad_mag[i,j-1]==255)
                   or (i!=rows-1 and j!=cols-1 and grad_mag[i+1,j+1]==255) 
                   or (i!=0 and j!=0 and grad_mag[i-1,j-1]==255)):
                    grad_mag[i,j]=255
                else:
                    grad_mag[i,j]=0
    return grad_mag

"""Canny edge detector which calls the above functions in order
img: Image
gaussian_ksize: Gaussian kernel size for denoising
sobel_ksize: Sobel kernel size for gradient computation
t: Lower value of threshold
T: Upper value of threshold
"""
def canny_edge_detector(img,gaussian_ksize,sobel_ksize,t,T):
    img = gaussian_blur(img,gaussian_ksize)
    grad_x,grad_y,grad_mag,grad_dir =  intensity_gradient(img,sobel_ksize)
    grad_mag = non_max_suppression(grad_mag,grad_dir)
    edges = hysterisis_thresholding(grad_mag,t,T)
    return edges

"""Outputs a colored image with color intensity based on gradient magnitude and value based on 
gradient direction computed in HSV colorspace
edges: Binary edge image after canny edge detection
grad_mag: Gradient magnitude returned by intensity_gradient
grad_dir: Rounded off gradient_direction
"""
def output_image(edges,grad_mag,grad_dir):
    rows,cols = edges.shape
    res_img = np.zeros((rows,cols,3), dtype = np.uint8) 
    max_grad = np.max(grad_mag)
    min_grad = np.min(grad_mag)
    for i in range(rows):
        for j in range(cols):
            if(edges[i][j]):
                v = int(255*((grad_mag[i][j] - min_grad)/(max_grad - min_grad)))
                if(grad_dir[i][j] == 0) :
                    res_img[i][j] = [0,255,v]
                elif(grad_dir[i][j] == 45) :
                     res_img[i][j] = [45,255,v]
                elif(grad_dir[i][j] == 90) :
                     res_img[i][j] = [90,255,v]
                else :
                     res_img[i][j] = [135,255,v]
                        
    color_img = cv2.cvtColor(res_img,cv2.COLOR_HSV2RGB)
    return color_img

