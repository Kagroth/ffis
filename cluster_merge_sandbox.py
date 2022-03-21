from sklearn import cluster
from fcm_analyzer import FCMAnalyzer
from utils import load_dataset, split_dataset, remove_outliers
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import linkage
import numpy as np

number_of_models = 3

data, feature_names = load_dataset("winequality-red.csv")
data = remove_outliers(data, neighbors=3)
print(data.shape)
np.random.shuffle(data)
data = data[:300, :]
data = StandardScaler().fit_transform(data)
# datasets = split_dataset(data, number_of_models)
datasets = split_dataset(data, number_of_models)
# data = data[:30, :]
cluster_centers = np.zeros((0, 12))

i = 1
for d in datasets:
    fcm_analyzer = FCMAnalyzer()
    clustering = fcm_analyzer.fit(d.T, error=0.001, maxiter=100)
    # cc = fcm_analyzer.get_partition(k=5)['cluster_centers']
    cc = fcm_analyzer.get_best_partition()['cluster_centers']
    print("Clusters count of ", i, " FCM")
    print(cc.shape[0])
    i = i + 1
    # cluster_centers.append(fcm_analyzer.get_partition(k=2)['cluster_centers'])
    cluster_centers = np.vstack((cluster_centers, cc))

print("========================================")
print(cluster_centers.shape)
print(cluster_centers)
# clustering = AgglomerativeClustering().fit(data)
print("Total number of clusters: ", cluster_centers.shape[0])
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1).fit(cluster_centers)
# clustering = linkage(cluster_centers)
# print(clustering)
print(clustering.labels_)
print("Clusters after agglomerative merge: ", clustering.n_clusters_)
print("==========================")
print("Euclidean distance")
dist_mtrx = distance_matrix(cluster_centers, cluster_centers)
print(dist_mtrx)
print(dist_mtrx.min())
zero_indices = np.argwhere(dist_mtrx == 0)
for index in zero_indices:
    x, y = index[0], index[1]
    dist_mtrx[x, y] = np.Inf
print(dist_mtrx)
print(dist_mtrx.min())
print(np.argwhere(dist_mtrx == dist_mtrx.min()))
print(np.sort(dist_mtrx, axis=None))