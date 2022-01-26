
from random_data_generator import RandomDataGenerator
from fcm_analyzer import FCMAnalyzer
from fcm_visualizer import FCMVisualizer
from utils import create_fuzzy_variables_from_clusters, create_rules_from_clusters
from tskmodel import TSKModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os
import time
import FuzzySystem as fuzz
import numpy as np
import matplotlib.pyplot as plt

rdg = RandomDataGenerator()
fcm_analyzer = FCMAnalyzer()

""" 
# 3D normal points
feature_names = ['X', 'Y', 'Z']
points1 = rdg.get_3D_normal_points(size=5, means=(0, 0, 0), stds=(1, 1, 1))
points2 = rdg.get_3D_normal_points(size=5, means=(10, 0, 0), stds=(1, 1, 1))
points3 = rdg.get_3D_normal_points(size=5, means=(0, 10, 0), stds=(1, 1, 1))
points4 = rdg.get_3D_normal_points(size=5, means=(0, 0, 10), stds=(1, 1, 1))
points5 = rdg.get_3D_normal_points(size=5, means=(10, 10, 0), stds=(1, 1, 1))
data = np.hstack((points1, points2, points3, points4, points5)).T
 """

""" 
# 3D uniform points
feature_names = ['X', 'Y', 'Z']
points1 = rdg.get_3D_uniform_points(size=10, value_range=(3, 7))
points2 = rdg.get_3D_uniform_points(size=10, value_range=(2, 8))
points3 = rdg.get_3D_uniform_points(size=10, value_range=(0, 5))
points4 = rdg.get_3D_uniform_points(size=10, value_range=(1, 3))
points5 = rdg.get_3D_uniform_points(size=10, value_range=(1, 2))
data = np.hstack((points1, points2, points3, points4, points5)).T """

""" # G datapoints from K. Passino
feature_names = ['X', 'Y', 'Z']
data = np.array([[0, 2, 1], [2, 4, 5], [3, 6, 6], [1, 1, 2]])
print(data) """

# Wine Quality dataset (red wine)
dataset_path = os.path.join(os.getcwd(), "datasets", "winequality-red.csv")
feature_names = list()
with open(dataset_path, "r") as f:
    headers = f.readline()
    headers_list = headers.split(";")
    headers_list = [header.replace("\"", "").replace("\n", "") for header in headers_list]
    feature_names = headers_list
print(feature_names)
data = np.genfromtxt(os.path.join(os.getcwd(), "datasets", "winequality-red.csv"), delimiter=";", skip_header=True)
print(data)

# scaler = MinMaxScaler()
scaler = StandardScaler()
data_min_max = scaler.fit_transform(data[:100, :]).T

clustering_result = fcm_analyzer.fit(data_min_max, error=0.005, maxiter=1000)
fcm_analyzer.show_fpc()

# use for 3D data
# fcm_visualizer = FCMVisualizer(fcm_clustering_result=clustering_result)
# fcm_visualizer.view_all_partitions()
# #########################################################################

best_fuzzy_partition = fcm_analyzer.get_best_partition()
# best_fuzzy_partition = fcm_analyzer.get_partition(k=3)

fuzzy_vars = create_fuzzy_variables_from_clusters(best_fuzzy_partition['cluster_centers'], feature_names=feature_names, show_fuzzy_vars=False)

# Show normalized fuzzy variable membership functions
# for fuzzy_var in fuzzy_vars:
#     fuzzy_var.show()

rules = create_rules_from_clusters(best_fuzzy_partition['cluster_centers'], fuzzy_vars)

for r in rules:
    r.show()

tsk_model = TSKModel(rules, epochs=200, lr=0.01)

start_time = time.time()

error_history = tsk_model.fit(data_min_max.T[:, :-1], feature_names[:-1], data_min_max.T[:, -1])

end_time = time.time()

print("Error: ", error_history[-1])
print("Learning time: {} s".format(end_time - start_time))

plt.plot(error_history)

plt.show()