#Contains utilities used in common
from matplotlib import pyplot as plt

#Displays grayscale image
def display_grayscale(img):
    plt.imshow(img,cmap="gray")
    plt.show()

#Displays image in RGB colorspace
def display_rgb(img):
    plt.imshow(img)
    plt.show()