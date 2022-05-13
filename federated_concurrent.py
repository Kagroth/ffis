from numpy import number
from federated_model import FederatedModel
from utils import load_dataset, split_dataset, remove_outliers, make_missing_values, knn_impute_dataset
from local_model import LocalModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import multiprocessing
import matplotlib.pyplot as plt

def run_model(model):
    model.fit()
    return model

if __name__ == '__main__':
    number_of_models = 3

    data, feature_names = load_dataset("winequality-red.csv")
    # data, feature_names = load_dataset("winequality-white.csv")
    data = remove_outliers(data, neighbors=3)
    print(data.shape)
    data = data[:30, :]
    
    datasets = split_dataset(data, number_of_models)
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

    # exit()
    local_models = list()
    training_times = list()

    for i in range(number_of_models):
        lm = LocalModel(datasets[i], datasets_weights[i], feature_names, epochs=10, validity_method="Spearman") 
        local_models.append(lm)

    local_rules = list()

    # for lm in local_models:
    #     rules = lm.create_rules(StandardScaler())
    #     local_rules += rules

    # print("Fed rules number:  ", len(local_rules))

    # for lm in local_models:
    #     lm.set_rules(local_rules)

    # m = FederatedModel.run_model(local_models[0])
    # print(m)
    # exit()
    fm = FederatedModel(local_models, feature_names, rounds_count=1)
    rules = fm.create_rules()
    # print(rules)
    # for r in rules:
    #     r.show()
    fm.set_federated_rules_to_local_models(rules)
    fm.fit()
    exit()
    with multiprocessing.Pool(number_of_models) as p:
        st = time.time()
        # local_models = p.map(run_model, [lm for lm in local_models])
        local_models = p.map(FederatedModel.run_model, [lm for lm in local_models])
        et = time.time()
        print("Time: {} s".format(et-st))

    i = 0

    for lm in local_models:
        print("=" * 200)
        print("Model nr {:2d} \
            Training error: {:1.6f} \
            Testing error: {:1.6f}".format(i + 1, lm.error_history[-1], lm.test_mse))
        print("=" * 200)
        i = i + 1

    fm = FederatedModel()
    fm.merge_local_models(local_models)

    print("")
    print("=" * 100)
    print("Federated model: ")
    i = 0
    # test local models on local datasets with federated model
    for lm in local_models:
        lm.set_fuzzy_inference_system(fm.ffis)
        mse = lm.test()
        model_str = ""
        # mse is federated mse, lm.test_mse is local model mse
        if mse > lm.test_mse:
            model_str = "Local"
        else:
            model_str = "Federated"

        print("=" * 100)
        print("Dataset nr {:2d} \
            Testing error: {:1.6f} \
            Better model: {:12s}".format(i + 1, mse, model_str))
        print("=" * 100)
        i = i + 1


