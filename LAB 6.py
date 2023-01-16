# 6. Clustering algorithms for unsupervised classification.

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
df.head()
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.show()
model = KMeans(n_clusters=5)
model.fit(X)
cluster_number = model.predict(X)
len(cluster_number)

c0 = X[cluster_number == 0]
c1 = X[cluster_number == 1]
c2 = X[cluster_number == 2]
c3 = X[cluster_number == 3]
c4 = X[cluster_number == 4]

plt.scatter(c0['Annual Income (k$)'], c0['Spending Score (1-100)'], c='red')
plt.scatter(c1['Annual Income (k$)'], c1['Spending Score (1-100)'], c='blue')
plt.scatter(c2['Annual Income (k$)'], c2['Spending Score (1-100)'], c='yellow')
plt.scatter(c3['Annual Income (k$)'], c3['Spending Score (1-100)'], c='cyan')
plt.scatter(c4['Annual Income (k$)'], c4['Spending Score (1-100)'], c='green')
plt.show()
