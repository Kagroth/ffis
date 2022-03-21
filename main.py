
from random_data_generator import RandomDataGenerator
from fcm_analyzer import FCMAnalyzer
from utils import feature_std, create_fuzzy_variables_from_clusters, create_rules_from_clusters, load_dataset, split_dataset, remove_outliers
from tskmodel import TSKModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import os
import time
import numpy as np
import matplotlib.pyplot as plt

rdg = RandomDataGenerator()
fcm_analyzer = FCMAnalyzer()

# Wine Quality dataset (red wine)
# dataset_path = os.path.join(os.getcwd(), "datasets", "winequality-red.csv")
# feature_names = list()
# with open(dataset_path, "r") as f:
#     headers = f.readline()
#     headers_list = headers.split(";")
#     headers_list = [header.replace("\"", "").replace("\n", "") for header in headers_list]
#     feature_names = headers_list

# data = np.genfromtxt(dataset_path, delimiter=";", skip_header=True)

data, feature_names = load_dataset("winequality-red.csv")
print(data)
print(feature_names)
# exit()
# datasets = split_dataset(data, 3)
# for d in datasets:
#     print(d.shape)
# print(datasets[0])
# exit()
# scaler = MinMaxScaler()
print("With outliers: ", data.shape)
data = remove_outliers(data, neighbors=3)
print("Without outliers: ", data.shape)
# exit()
scaler = StandardScaler()
data_min_max = scaler.fit_transform(data[:, :])
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(data_min_max[:, :-1], data_min_max[:, -1], test_size=0.2)


clustering_result = fcm_analyzer.fit(data_min_max.T, error=0.001, maxiter=100)
fcm_analyzer.show_fpc()
part = fcm_analyzer.get_partition(k=2)
best_fuzzy_partition = fcm_analyzer.get_best_partition()
# best_fuzzy_partition = fcm_analyzer.get_partition(k=5) # use for select k-partition

fuzzy_vars = create_fuzzy_variables_from_clusters(best_fuzzy_partition['cluster_centers'], 
                                                cluster_stds=best_fuzzy_partition['crisp_cluster_stds'], 
                                                feature_names=feature_names, 
                                                show_fuzzy_vars=False)

rules = create_rules_from_clusters(best_fuzzy_partition['cluster_centers'], fuzzy_vars)

for r in rules:
    r.show()

tsk_model = TSKModel(rules, epochs=500, lr=0.2, momentum=0.9, eg=0.05)

start_time = time.time()

error_history = tsk_model.fit(train_inputs, feature_names[:-1], train_outputs)

end_time = time.time()
test_mse = tsk_model.test(test_inputs, test_outputs)

print("Error: ", error_history[-1])
print("Learning time: {} s".format(np.round(end_time - start_time, decimals=2)))
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.plot(error_history)


print("Training MSE: ", error_history[-1])
print("Testing MSE: ", test_mse)

print("Clustering fpc:")
fcm_analyzer.show_fpc()
print("")

for r in tsk_model.fis.rules:
    r.show()
    print("")

for index, rule in enumerate(tsk_model.fis.rules):
    print("Coefficients for rule {}: {} \n".format(index+1, rule.consequent.get_params()))

for fuzzy_var_name, fuzzy_var in tsk_model.fis.rules[0].antecedent.fuzzy_variables.items():
    fuzzy_var.show()

plt.show()