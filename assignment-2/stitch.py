import cv2
import numpy as np
from matplotlib import pyplot as plt
from harris_corner_detector import *

def display_image(img):
    plt.imshow(img,cmap="gray")
    plt.show()

def gaussian_blur(img,kx,ky):
    blurred_img = cv2.GaussianBlur(img,(kx,ky),0)
    return blurred_img

def get_transformation_matrix(points1,points2):
    assert(points1.shape[0]==points2.shape[0])
    A = np.zeros((8,9))
    for i in range(4):
        x,y = points1[i][0],points1[i][1]
        u,v = points2[i][0],points2[i][1]
        A[2*i] = [-x,-y,-1,0,0,0,u*x,u*y,u]
        A[2*i+1] = [0,0,0,-x,-y,-1,v*x,v*y,v]
    U,S,V = np.linalg.svd(A)
    H = V[8].reshape((3,3))        
    return H

def get_dimensions(H,img1,img2):
    x_co_ords = []
    y_co_ords = []
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    h2 = img2.shape[0]
    w2 = img2.shape[1]
    corners = np.array([[[0],[0]],[[h1-1],[0]],[[h1-1],[w1-1]],[[0],[w1-1]]])
    for i in range(4):
        p = corners[i]
        p = np.append(p,[1])
        p = np.dot(H,p)
        x_co_ords.append(p[0]/p[2])
        y_co_ords.append(p[1]/p[2])
    
    min_x = int(np.min(x_co_ords))
    max_x = int(np.max(x_co_ords))
    min_y = int(np.min(y_co_ords))
    max_y = int(np.max(y_co_ords))

    res_w = int(max_y-min_y)+w2
    res_h = int(max_x-min_x)

    return res_h,res_w,min_x,max_x,min_y,max_y

def plotCommonCorners(points, image) :
    num = points.shape[0]
    for i in range(num) :
        for j in range(5):
            for k in range(5):
                image.itemset((points[i][0]+j,points[i][1]+k,0),255)
                image.itemset((points[i][0]+j,points[i][1]+k,1),0)
                image.itemset((points[i][0]+j,points[i][1]+k,2),0)
    return image

def stitch(H,img1,img2,color_img1,color_img2):
    res_h,res_w,min_x,max_x,min_y,max_y = get_dimensions(H,img1,img2)
    res = np.zeros((res_h,res_w,3),dtype=np.uint8)

    #print("Min x and Min y\n",min_x,min_y)
    #Apply transformation matrix to each co-ordinate for projection to the plane of the second image
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    h2 = img2.shape[0]
    w2 = img2.shape[1]
    for i in range(h1):
        for j in range(w1):
            p = np.array([[i],[j],[1]])
            p = np.dot(H,p)
            x1 = int(p[0]/p[2])-min_x
            y1 = int(p[1]/p[2])-min_y

            if x1>0 and y1>0 and y1<max_y-min_y and x1<max_x-min_x:
                        res[x1,y1]=color_img1[i,j]
    
    plt.imsave('outputs/perspective.png',res)
    for i in range(h2):
        for j in range(w2):
            if (i-min_x)> 0 and (i-min_x)<res_h and (j-min_y)>0 and (j-min_y)<res_w and np.array_equal(res[i-min_x][j-min_y],np.array([0,0,0])):
                res[i-min_x][j-min_y]=color_img2[i][j]
    
    return res,min_x,min_y

