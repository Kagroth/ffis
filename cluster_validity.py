from skfuzzy.cluster import cmeans
from sklearn import cluster
from utils import load_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

CLUSTER_VALIDITY_METHODS = ["Pearson", "Spearman", "fpc"]

class CorrelationClusterValidity:
    def __init__(self, data) -> None:
        self.name = None
        self.data = data
        self.distance_matrix = None
    
    def create_distance_matrix(self, normalize=True) -> None:
        self.distance_matrix = pairwise_distances(self.data, self.data)

        if normalize:
            self.distance_matrix = self.distance_matrix / np.max(self.distance_matrix)
        
    def create_dissimilarity_matrix(self, membership_matrix) -> np.ndarray:
        # calculate UTU matrix
        utu = np.zeros(self.distance_matrix.shape)
        n = self.distance_matrix.shape[0]
        clusters_count = membership_matrix.shape[1]

        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(clusters_count):
                    s += membership_matrix[i, k] * membership_matrix[j, k]
                utu[i, j] = s
        
        utu_norm = utu / np.max(utu)
        dissimilarity_mtrx = np.ones(self.distance_matrix.shape) - utu_norm

        return dissimilarity_mtrx

class PearsonCCV(CorrelationClusterValidity):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.name = "Pearson CCV"

    def compute(self, membership_matrix) -> float:
        self.create_distance_matrix()
        diss_matrix = self.create_dissimilarity_matrix(membership_matrix)
        # Pearson Correlation cluster validity
        data_distance_mean = np.full(self.distance_matrix.shape, np.mean(self.distance_matrix))
        diss_matrix_mean = np.full(diss_matrix.shape, np.mean(diss_matrix))
        n = self.distance_matrix.shape[0]

        numerator = 0
        denominator_left = 0
        denominator_right = 0

        for i in range(n):
            for j in range(n):
                numerator += (self.distance_matrix[i, j] - data_distance_mean[i, j]) * (diss_matrix[i, j] - diss_matrix_mean[i, j])
                denominator_left += (self.distance_matrix[i, j] - data_distance_mean[i, j]) * (self.distance_matrix[i, j] - data_distance_mean[i, j])
                denominator_right += (diss_matrix[i, j] - diss_matrix_mean[i, j]) * (diss_matrix[i, j] - diss_matrix_mean[i, j]) 

        denominator_left = np.sqrt(denominator_left)
        denominator_right = np.sqrt(denominator_right)

        denominator = denominator_left * denominator_right

        result = numerator / denominator

        return result

class SpearmanCCV(CorrelationClusterValidity):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.name = "Spearman CCV"

    def compute(self, membership_matrix) -> float:
        self.create_distance_matrix()
        diss_matrix = self.create_dissimilarity_matrix(membership_matrix)
        # Spearmans correlation cluster validity
        ccvs_sum = 0
        r = 0
        n = self.distance_matrix.shape[0]

        squared = np.square(self.distance_matrix - diss_matrix)
        ccvs_sum = np.sum(squared) - np.sum(squared[np.diag_indices(n)])

        p = ccvs_sum * 6
        nnn = (n * n * n) - n
        r = p / nnn

        ccvs_index = 1 - r
        
        return ccvs_index

# ************************************************************
# data generation
# Horizontal lines data
# points_count = 33
# data_x = np.random.uniform(low=-6, high=6, size=points_count)
# data_y = np.random.uniform(low=7, high=9, size=points_count)
# hldata1 = np.column_stack((data_x, data_y))

# data_x = np.random.uniform(low=-6, high=6, size=points_count)
# data_y = np.random.uniform(low=-1, high=1, size=points_count)
# hldata2 = np.column_stack((data_x, data_y))

# data_x = np.random.uniform(low=-6, high=6, size=points_count)
# data_y = np.random.uniform(low=-9, high=-7, size=points_count)
# hldata3 = np.column_stack((data_x, data_y))

# data = np.vstack((hldata1, hldata2, hldata3))

# data = np.asarray([[2, 3], [6, 2], [4, 4]])

# data1 = np.random.normal(loc=(0, 0), scale=1, size=(100, 2))
# data2 = np.random.normal(loc=(10, 10), scale=1, size=(100, 2))
# data3 = np.random.normal(loc=(0, 10), scale=1, size=(100, 2))
# data = np.vstack((data1, data2, data3))

# data, feature_names = load_dataset("winequality-red.csv")
# np.random.shuffle(data)
# data = data[:600, :]
# data = StandardScaler().fit_transform(data)

# ccvs_results = dict()
# ccvp_results = dict()
# cntr = None

# ccvp_o = PearsonCCV(data)
# ccvs_o = SpearmanCCV(data)

# for clusters_n in range(2, 10):
#     print()
#     print("=" * 100)
#     print("Clusters count: ", clusters_n)
#     cntr, u, u0, d, jm, p, fpc = cmeans(data.T, clusters_n, 2, error=0.01, maxiter=1000)
#     u = u.T

#     ccvp_value = ccvp_o.compute(u)
#     ccvs_value = ccvs_o.compute(u)

#     print("Spearmans CCV: ", ccvs_value)
#     ccvs_results[clusters_n] = ccvs_value

#     print("Pearson CCV: ", ccvp_value)
#     ccvp_results[clusters_n] = ccvp_value
    
# print("Pairs that should be compared: ", data.shape[0] * (data.shape[0] - 1) / 2)
# for key, value in ccvs_results.items():
#     print("Clusters count: {}, ccvs: {}, ccvp: {}".format(key, value, ccvp_results[key]))
# print("Best ccvs: ", max(ccvs_results, key=ccvs_results.get))
# print("Best ccvp: ", max(ccvp_results, key=ccvp_results.get))