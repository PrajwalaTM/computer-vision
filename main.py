from utils import display_grayscale
from utils import display_rgb
from canny_edge_detector import *
from harris_corner_detector import *
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='file',help='canny_edge/harris_corner')

canny_parser=subparsers.add_parser('canny_edge',help='Canny edge detector')
canny_parser.add_argument('source',metavar='src',help='Input image (jpg,png)')
canny_parser.add_argument('gaussian_ksize',help='Gaussian kernel size')
canny_parser.add_argument('sobel_ksize',help='Sobel kernel size')
canny_parser.add_argument('t',help='Lower threshold')
canny_parser.add_argument('T',help='Upper threshold')

harris_corner=subparsers.add_parser('harris_corner',help='Harris corner detector')
harris_corner.add_argument('source',metavar='src',help='Input image (jpg,png)')
harris_corner.add_argument('window_size',help='Window size')
harris_corner.add_argument('sobel_ksize',help='Sobel kernel size')
harris_corner.add_argument('k',help='Harris corner detector parameter')
harris_corner.add_argument('t',help='Response threshold')

args = parser.parse_args()

if args.file=='canny_edge':
    img = cv2.imread(args.source,0)
    gaussian_ksize = int(args.gaussian_ksize)   
    sobel_ksize = int(args.sobel_ksize)
    t = int(args.t)
    T = int(args.T)
    
    edges = canny_edge_detector(img,gaussian_ksize,sobel_ksize,t,T)
    grad_x,grad_y,grad_mag,grad_dir=intensity_gradient(img,sobel_ksize)
    
    plt.figure(num="Gradients in X and Y directions")
    plt.subplot(211)
    plt.imshow(np.abs(grad_x),cmap="gray")
    plt.subplot(212)
    plt.imshow(np.abs(grad_y),cmap="gray")
    plt.show()

    display_grayscale(edges)
    res_img = output_image(edges,grad_mag,grad_dir)
    display_rgb(res_img)
    plt.imsave('outputs/edges_image',res_img)

elif args.file=='harris_corner':
    img = cv2.imread(args.source,0)
    window_size = int(args.window_size)
    sobel_ksize = int(args.sobel_ksize)
    k = float(args.k)
    t = int(args.t)
    res_img,corner_points = harris_corner_detector(img,window_size,sobel_ksize,k,t)
    
    print 'Number of corner points detected = ',len(corner_points)
    display_grayscale(res_img)
    plt.imsave('outputs/corner_points_image',res_img)

else:
    pass