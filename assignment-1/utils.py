#Contains utilities used in common
from matplotlib import pyplot as plt
import numpy as np

#Displays grayscale image
def display_grayscale(img):
    plt.imshow(img,cmap="gray")
    plt.show()

#Displays image in RGB colorspace
def display_rgb(img):
    plt.imshow(img)
    plt.show()

#Evaluates performance metrics - Accuracy and F Score
def performance_metrics(ideal,actual):
    ideal_neg = 1-ideal
    actual_neg = 1-actual
    tp,tn,fp,fn = (0.0,0.0,0.0,0.0)
    assert(ideal.shape==actual.shape)
    tp = ((np.logical_and(ideal,actual))==1).sum()
    fp = (actual==1).sum()-tp
    tn = ((np.logical_and(ideal_neg,actual_neg))==1).sum()
    fn = (actual_neg==1).sum() - tn
    acc = 1.0*(tp + tn)/(tp+fn+tn+fp) # Accuracy
    fsc = 1.0*(2*tp)/((2*tp)+fp+fn) # F-score
    return (acc,fsc)