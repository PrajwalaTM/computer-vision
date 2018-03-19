import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from utils import *

def extract_sift_features(image):
    sift_object = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift_object.detectAndCompute(image, None)
    return [keypoints, descriptors]

def kmeans(descriptor_list,num_clusters):
    vStack = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_vstack = vStack.copy()
    k_means_obj = KMeans(n_clusters=num_clusters,n_jobs=-1)
    k_means_ret = k_means_obj.fit_predict(descriptor_vstack)
    return [k_means_obj,k_means_ret]

def develop_bag_of_words(descriptor_list,k_means_ret,num_clusters,num_images):
    final_histogram = np.array([np.zeros(num_clusters) for i in range(num_images)])
    old_count = 0
    for i in range(num_images):
        l = len(descriptor_list[i])
        for j in range(l):
            idx = k_means_ret[old_count+j]
            final_histogram[i][idx] += 1
        old_count += l
    print "Vocabulary Histogram Generated"
    return final_histogram

def preprocessing(final_histogram):
    scale = StandardScaler().fit(final_histogram)
    final_histogram = scale.transform(final_histogram)
    return [scale,final_histogram]

def train(final_histogram,train_labels):
    clf = GaussianNB()
    clf.fit(final_histogram,train_labels)
    print("Training completed")
    return clf

def recognize_image(clf,scale,k_means_obj,test_image,NUM_CLUSTERS):
    kp,desc = extract_sift_features(test_image)
    vocab = np.array( [[ 0 for i in range(NUM_CLUSTERS)]])
    k_means_test_ret = k_means_obj.predict(desc)
    for each in k_means_test_ret:
            vocab[0][each] += 1
    print(vocab)
    vocab = scale.transform(vocab)
    label = clf.predict(vocab)
    return label

def train_model(train_path,NUM_CLUSTERS):
    images, image_count = get_images(train_path)
    print(image_count)
    label_count = 0
    train_labels = np.array([])
    label_encoding = {}
    descriptor_list = []
    for label,image_list in images.iteritems():
        label_encoding[str(label_count)] = label
        print("Computing features for ",label)
        for img in image_list:
            train_labels = np.append(train_labels,label_count)
            kp,desc = extract_sift_features(img)
            descriptor_list.append(desc)
        
        label_count+=1
    print("Descriptor list size",len(descriptor_list))
    k_means_obj,k_means_ret = kmeans(descriptor_list=descriptor_list,num_clusters=NUM_CLUSTERS)
    np.save('k_means_ret.npy',k_means_ret)
    num_images = len(descriptor_list)
    final_histogram = develop_bag_of_words(descriptor_list=descriptor_list,k_means_ret=k_means_ret,num_clusters=NUM_CLUSTERS,num_images=num_images)
    scale,final_histogram = preprocessing(final_histogram)
    np.save('bag_of_words.npy',final_histogram)
    clf= train(final_histogram,train_labels)
    return [clf,final_histogram,label_encoding,k_means_obj]

def test_model(clf,k_means_obj,test_path,NUM_CLUSTERS):
    images, image_count = get_images(test_path)
    y_true = []
    y_predicted = []
    for label,image_list in images.iteritems():
        print("Processing ",label)
        for img in image_list:
            predicted_label = recognize_image(clf,scale,k_means_obj,img,NUM_CLUSTERS)
            print(predicted_label)
            y_true.append(label)
            y_predicted.append(label_encoding[str(int(predicted_label[0]))])
    print(len(y_true))
    print(len(y_predicted))
    score = accuracy_score(y_true,y_predicted)
    print("Accuracy ",score)
    conf_matrix = confusion_matrix(y_true,y_predicted)
    print("Confusion matrix ",conf_matrix)
