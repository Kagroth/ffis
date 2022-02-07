from skfuzzy.cluster import cmeans
from utils import feature_std
import numpy as np

class FCMAnalyzer:
    def __init__(self, clusters=None) -> None:
        if clusters is None:
            self.clusters = [x for x in range(2, 10)]
        elif isinstance(clusters, list):
            if 0 in clusters or 1 in clusters:
                print("k cannot be equal 0 or 1")
                return
                
            self.clusters = clusters
        else:
            print("Error! Invalid type of clusters parameter")
        

    def fit(self, data, error=0.01, maxiter=10) -> list:
        # k can't be bigger than number of data points
        if data.shape[1] <= max(self.clusters):
            self.clusters = [x for x in range(2, data.shape[1] + 1)]

        self.clustering_result = []

        for cluster_count in self.clusters:
            cntr, u, u0, d, jm, p, fpc = cmeans(data, cluster_count, 2, error=error, maxiter=maxiter, init=None)
            
            clusters = list() # list for cluster members
            cluster_indices = list() # list for indices for cluster with highest membership 

            # init k empty lists for cluster members
            for index in range(cluster_count):
                clusters.append(list())

            # get maximum membership value for every data point
            max_of_each_column = np.max(u, axis=0)

            # find the index of cluster with highest membership for every data point
            for index in range(len(max_of_each_column)):
                index_of_cluster_for_vector = np.where(u[:, index] == max_of_each_column[index])
                cluster_indices.append(index_of_cluster_for_vector[0][0])

            # Fill the clusters list with data points with highest membership values
            for col_index in range(u.shape[1]):
                vector = data[:, col_index]
                cluster_index = cluster_indices[col_index]
                
                # if clusters[cluster_index] is None:
                    # clusters[cluster_index] = np.array()
                
                clusters[cluster_index].append(vector.tolist())

            crisp_cluster_stds = list()
            for cluster in clusters:
                crisp_cluster_stds.append(feature_std(cluster))

            self.clustering_result.append({
                "k": cluster_count,
                "fpc": fpc,
                "cluster_centers": cntr,
                "membership": {"points": data, "u": u},
                "crisp_clusters": clusters,
                "crisp_cluster_stds": np.array(crisp_cluster_stds)
            })
        
        return self.clustering_result
    
    def show_fpc(self) -> None:
        for c_r in self.clustering_result:
            print("Fpc of clustering for k = {}; {}".format(c_r["k"], c_r["fpc"]))

    def get_best_partition(self) -> object:
        max_fpc_index = 0
        max_fpc = self.clustering_result[max_fpc_index]['fpc']

        for index in range(len(self.clustering_result)):
            if self.clustering_result[index]['fpc'] > max_fpc:
                max_fpc_index = index
                max_fpc = self.clustering_result[index]['fpc']
                
        return self.clustering_result[max_fpc_index]
    
    def get_partition(self, k) -> object:
        for index in range(len(self.clustering_result)):
            if self.clustering_result[index]['k'] == k:
                return self.clustering_result[index]
        
        # if there is no k-partition, return best partition
        return self.get_best_partition()