from skfuzzy.cluster import cmeans
from utils import feature_std
from cluster_validity import CLUSTER_VALIDITY_METHODS, PearsonCCV, SpearmanCCV
import numpy as np

class FCMAnalyzer:
    def __init__(self, clusters=None, validity_method=None) -> None:
        self.validity_method = None # validity method name
        self.validity_index = None # validity method object

        if clusters is None:
            self.clusters = [x for x in range(2, 10)]
        elif isinstance(clusters, list):
            if 0 in clusters or 1 in clusters:
                print("k cannot be equal 0 or 1")
                return
                
            self.clusters = clusters
        else:
            print("Error! Invalid type of clusters parameter")
        
        if validity_method is None:
            validity_method = "fpc"
        elif validity_method in CLUSTER_VALIDITY_METHODS:
            self.validity_method = validity_method
        else:
            print("Error! Invalid validity method")
        

    def fit(self, data, error=0.01, maxiter=10) -> list:
        if self.validity_method == "Pearson":
            self.validity_index = PearsonCCV(data.T) # data is CxN because of cmeans, so there is need to transpose it
        elif self.validity_method == "Spearman":
            self.validity_index = SpearmanCCV(data.T)
        else:
            self.validity_index = None

        # k can't be bigger than number of data points
        if data.shape[1] <= max(self.clusters):
            self.clusters = [x for x in range(2, data.shape[1] + 1)]

        self.clustering_result = []

        for cluster_count in self.clusters:
            cntr, u, u0, d, jm, p, fpc = cmeans(data, cluster_count, 2, error=error, maxiter=maxiter, init=None)
            # print("Membership matrix for k = ", cluster_count)
            # print(u.T.shape)
            # print(u.T)

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
                # print("Cluster members count: ", len(cluster))
                if len(cluster) == 0 or len(cluster) == 1:
                    crisp_cluster_stds.append(np.ones((data.T.shape[1])))
                    continue

                crisp_cluster_stds.append(feature_std(cluster))

            validity_value = self.validity_index.compute(u.T)
            # print(crisp_cluster_stds)
        
            # cluster_stds = feature_std(data.T)
            # print("CLuster stds on U")
            # print(cluster_stds)

            self.clustering_result.append({
                "k": cluster_count,
                "fpc": fpc,
                "cluster_centers": cntr,
                "membership": {"points": data, "u": u},
                "crisp_clusters": clusters,
                # "crisp_cluster_stds": cluster_stds,
                "crisp_cluster_stds": np.array(crisp_cluster_stds),
                "validity_index": validity_value
            })
        
        return self.clustering_result
    
    def show_fpc(self) -> None:
        for c_r in self.clustering_result:
            print("Fpc of clustering for k = {}; {}".format(c_r["k"], c_r["fpc"]))

    def show_validity_indices(self) -> None:
        print("{} index: ".format(self.validity_index.name))

        for c_r in self.clustering_result:
            print("Validity index of clustering for k = {}; {}".format(c_r["k"], c_r["validity_index"]))

    def get_best_partition(self) -> object:
        max_validity_index = 0
        max_validity = self.clustering_result[max_validity_index]['validity_index'] 

        for index in range(len(self.clustering_result)):
            if self.clustering_result[index]['validity_index'] > max_validity:
                max_validity_index = index
                max_validity = self.clustering_result[index]['validity_index']
                
        return self.clustering_result[max_validity_index]
    
    def get_partition(self, k) -> object:
        for index in range(len(self.clustering_result)):
            if self.clustering_result[index]['k'] == k:
                return self.clustering_result[index]
        
        # if there is no k-partition, return best partition
        return self.get_best_partition()