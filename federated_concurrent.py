from federated_model import FederatedModel
from utils import load_dataset, split_dataset, split_dataset_grouped, remove_outliers, make_missing_values, knn_impute_dataset
from local_model import LocalModel

import matplotlib.pyplot as plt
import numpy as np

def run_model(model):
    model.fit()
    return model

if __name__ == '__main__':
    number_of_models = 2

    data, feature_names = load_dataset("winequality-red.csv")
    # data, feature_names = load_dataset("winequality-white.csv")
    data = remove_outliers(data, neighbors=3)
    print(data.shape)
    data = data[:200, :]
    
    # datasets = split_dataset(data, number_of_models)
    datasets = split_dataset_grouped(data, number_of_models)
    # datasets = split_dataset(data, number_of_models, overlap=True, block_size=200)
    
    print("Datasets shapes: ")
    shapes = ""
    for d in datasets:
        shapes += str(d.shape)
    print(shapes)
    index = 0
    datasets_weights = list()

    for i in range(number_of_models):
        datasets[i], d_weight = make_missing_values(datasets[i])
        datasets[i] = knn_impute_dataset(datasets[i])
        datasets_weights.append(d_weight)
        i += 1
    
    print("Datasets weights before normalization: ", datasets_weights)
    datasets_weights = np.array(datasets_weights) / sum(datasets_weights)
    print("Datasets weights after normalization: ", datasets_weights)
    
    local_models = list()
    training_times = list()

    for i in range(number_of_models):
        lm = LocalModel(datasets[i], datasets_weights[i], feature_names, epochs=100, validity_method="Spearman") 
        local_models.append(lm)

    fm = FederatedModel(local_models, feature_names, rounds_count=5)
    rules = fm.create_rules()

    fm.set_federated_rules_to_local_models(rules)
    fm.fit()
    print()
    print("Number of federated rules: ", fm.number_of_fed_rules)
    print()
    print("Dataset OWA weights")
    print(datasets_weights)
    print()
    print("Average of MSE on local and aggregated fis: ")
    print(np.mean(fm.final_results, axis=1))
    print()
    print("Average standard deviation on local and aggregated fis: ")
    print(np.std(fm.final_results, axis=1))
    exit()


