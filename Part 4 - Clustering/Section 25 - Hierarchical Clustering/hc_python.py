#Hierarchical Clustering

#Importng the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting hierarchial clustering to the mall data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y_hc = hc.fit_predict(X) 

#Visualising the clusters
plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1], s = 10, c = 'red', label = 'Careful' )
plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1], s = 10, c = 'blue', label = 'Standard' )
plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1], s = 10, c = 'green', label = 'Target' )
plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1], s = 10, c = 'cyan', label = 'Careless' )
plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1], s = 10, c = 'magenta', label = 'Sensible' )
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()