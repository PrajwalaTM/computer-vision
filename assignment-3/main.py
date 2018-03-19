import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from utils import *
from object_classifier import *

train_path = "/media/prajwala/40267F2A267F2058/Computer Engg/8th sem/computer-vision/assignment-3/train_data/"
test_path = "/media/prajwala/40267F2A267F2058/Computer Engg/8th sem/computer-vision/assignment-3/test_data/"
NUM_CLUSTERS = 120

clf,final_histogram,label_encoding,k_means_obj = train_model(train_path,NUM_CLUSTERS)
plotHist(NUM_CLUSTERS,final_histogram)
test_model(clf,k_means_obj,test_path,NUM_CLUSTERS)
