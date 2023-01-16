# 12. Plot the cluster data using python visualizations
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

data = load_digits().data
pca = PCA(2)
df = pca.fit_transform(data)

kmeans = KMeans(n_clusters=10)
label = kmeans.fit_predict(df)

filtered_label = df[label == 0]

plt.scatter(filtered_label[:, 0], filtered_label[:, 1])
plt.show()

filtered_label2 = df[label == 2]
filtered_label3 = df[label == 8]

plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color='red')
plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1], color='black')
plt.show()

u_label = np.unique(label)

for i in u_label:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=1)
plt.legend(u_label)
plt.show()

centroids = kmeans.cluster_centers_

for i in u_label:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=1)

plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.legend(u_label)
plt.show()
