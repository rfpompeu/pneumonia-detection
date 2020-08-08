import Data.data as dt

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import Classification_Algorithms.SVM.svc as svc


descriptors = dt.list_descriptors("train")
kmeans = KMeans(n_clusters=5).fit(descriptors)
dictionary = kmeans.cluster_centers_

x_train, y_train = dt.labels("train", dictionary)
x_test, y_test = dt.labels("test", dictionary)

prevision = svc.prevision(x_train, x_test, y_train, y_test)
print(accuracy_score(y_test, prevision))


def confusion_mat():
    matrix = confusion_matrix(y_test, prevision)
    matrix = (matrix/sum(matrix)).T
    matrix = pd.DataFrame(matrix, columns=['Normal', 'Pneumonia'], index=[
                          'Normal', 'Pneumonia'])
    sns.heatmap(matrix, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    plt.show()


confusion_mat()
