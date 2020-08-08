import Data.data as dt
from sklearn.cluster import KMeans
import numpy as np
import Classification_Algorithms.SVM.svc as svc

descriptors = dt.list_descriptors("train")
kmeans = KMeans(n_clusters=5).fit(descriptors)
dictionary = kmeans.cluster_centers_

x_train, y_train = dt.labels("train", dictionary)
x_test, y_test = dt.labels("test", dictionary)

print(svc.acurracy(x_train, x_test, y_train, y_test))