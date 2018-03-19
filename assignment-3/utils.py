import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

def get_images(path):
    image_list = {}
    image_count = 0
    for file in glob(path+"*"):
        category = file.split("/")[-1]
        print("Category ",category)
        image_list[category] = []
        for image in glob(path+category+"/*"):
            #print("File ",image)
            img = cv2.imread(image,0)
            image_list[category].append(img)
            image_count+=1
    return [image_list,image_count]

def plotHist(num_clusters,final_histogram):
    print "Plotting histogram"
    vocabulary = final_histogram
    x_scalar = np.arange(num_clusters)
    y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(num_clusters)])
    print y_scalar
    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()
    plt.savefig('histogram_'+str(num_clusters)+'.png')
