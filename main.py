import Data.data as dt
from sklearn.cluster import KMeans
import numpy as np


descriptors = dt.list_descriptors("test")
kmeans = KMeans(n_clusters=5).fit(descriptors)
dictionary = kmeans.cluster_centers_


x_train, y_train = dt.labels("test", dictionary)

print(x_train)
