from sklearn import cluster
from random_data_generator import RandomDataGenerator
from fcm_analyzer import FCMAnalyzer
from fcm_visualizer import FCMVisualizer
from utils import feature_std, create_fuzzy_variables_from_clusters, create_rules_from_clusters
from tskmodel import TSKModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

import os
import FuzzySystem as fuzz
import numpy as np
import matplotlib.pyplot as plt

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

# scaled_data = MinMaxScaler().fit_transform(data)
# scaled_data = StandardScaler().fit_transform(data)
scaled_data = QuantileTransformer(output_distribution="normal").fit_transform(data)
print(scaled_data)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
fcm_analyzer = FCMAnalyzer(clusters=[x for x in range(2, 13)])
clustering_result = fcm_analyzer.fit(principal_components[:, :].T, maxiter=100)
fcm_analyzer.show_fpc()
fcm_visualizer = FCMVisualizer(clustering_result)
fcm_visualizer.view_all_partitions()