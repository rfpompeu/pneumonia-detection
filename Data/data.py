import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def descriptors(filename):
    img = cv2.imread(filename, 0)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (9, 9), 1)
    orb = cv2.ORB_create(nfeatures=100)
    keydots = orb.detect(img, None)
    keydots, descriptors = orb.compute(img, keydots)
    return (descriptors)

def list_descriptors(type):
    values = np.empty((0, 32))
    for dir in ["NORMAL", "PNEUMONIA"]:
        for root, dirs, files in os.walk(os.path.join("Data", type, dir)):
            for filename in files:
                values = np.append(values, descriptors(
                    os.path.join("Data", type, dir, filename)), axis=0)
    return (values)

def labels(type, dictionary):
    x_train = []
    y_train = np.empty((0,), dtype=int)
    for dir in ["NORMAL", "PNEUMONIA"]:
        for root, dirs, files in os.walk(os.path.join("Data", type, dir)):
            for filename in files:
                alg_knn = NearestNeighbors(n_neighbors=1).fit(dictionary)
                closer = alg_knn.kneighbors(descriptors(os.path.join(
                    "Data", type, dir, filename)), return_distance=False).flatten()
                hist = np.histogram(
                    closer, bins=np.arange(dictionary.shape[0]+1))[0]
                x_train.append(hist.tolist())
                if(dir == "NORMAL"):
                    y_train = np.append(y_train, np.array([0]), axis=0)
                else:
                    y_train = np.append(y_train, np.array([1]), axis=0)
    x_train = np.array(x_train, dtype=int)
    return (x_train, y_train)