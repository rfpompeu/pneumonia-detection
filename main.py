import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def get_descriptors(way_img):
    h, w = 200, 200
    img = cv2.imread(way_img, 0)
    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (9, 9), 1)
    orb = cv2.ORB_create(nfeatures=100)
    keydots = orb.detect(img, None)
    keydots, descriptors = orb.compute(img, keydots)
    return descriptors


class WordPack:
    def gerator_dic(self, list_descr):
        kmeans = KMeans(n_clusters=50)
        kmeans = kmeans.fit(list_descr)
        self.dic = kmeans.cluster_centers_

    def hist(self, descr):
        alg_knn = NearestNeighbors(n_neighbors=1)
        alg_knn = alg_knn.fit(self.dic)
        closer = alg_knn.kneighbors(descr, return_distance=False).flatten()
        hist = np.histogram(closer, bins=np.arange(self.dic.shape[0]+1))[0]
        return hist


descriptors = np.empty((0, 32), dtype=np.uint8)
for dir in ["train/NORMAL", "train/PNEUMONIA"]:
    for root, dirs, files in os.walk(dir):
        for filename in files:
            descriptors = np.append(descriptors, get_descriptors(
                os.path.join(dir, filename)), axis=0)

wp = WordPack()
wp.gerator_dic(descriptors)

x_train_normal = np.array([])
x_train_pne = np.array([])

for dir in ["train/NORMAL", "train/PNEUMONIA"]:
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if(dir == "train/NORMAL"):
                x_train_normal = np.append(x_train_normal, wp.hist(
                    get_descriptors(os.path.join(dir, filename))))
            else:
                x_train_pne = np.append(x_train_pne, wp.hist(
                    get_descriptors(os.path.join(dir, filename))))


x_train_normal = x_train_normal.reshape((int(x_train_normal.shape[0]/50), 50))
y_train_normal = np.zeros(x_train_normal.shape[0])

x_train_pne = x_train_pne.reshape((int(x_train_pne.shape[0]/50), 50))
y_train_pne = np.ones(x_train_pne.shape[0])


x_train = np.append(x_train_normal, x_train_pne, axis=0)
y_train = np.append(y_train_normal, y_train_pne)


svc = svm.SVC(kernel="linear", C=0.8)
svc.fit(x_train, y_train)


x_test_normal = np.array([])
x_test_pne = np.array([])
for dir in ["test/NORMAL", "test/PNEUMONIA"]:
    for root, dirs, files in os.walk(dir):
        for filename in files:
            if(dir == "test/NORMAL"):
                x_test_normal = np.append(x_test_normal, wp.hist(
                    get_descriptors(os.path.join(dir, filename))))
            else:
                x_test_pne = np.append(x_test_pne, wp.hist(
                    get_descriptors(os.path.join(dir, filename))))

x_test_normal = x_test_normal.reshape((int(x_test_normal.shape[0]/50), 50))
y_test_normal = np.zeros(x_test_normal.shape[0])

x_test_pne = x_test_pne.reshape((int(x_test_pne.shape[0]/50), 50))
y_test_pne = np.ones(x_test_pne.shape[0])

x_test = np.append(x_test_normal, x_test_pne, axis=0)
y_test = np.append(y_test_normal, y_test_pne)

prevision = svc.predict(x_test)
accuracy = accuracy_score(y_test, prevision)

print("Accuracy:", accuracy)
