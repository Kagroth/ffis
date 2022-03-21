from numpy import number
from federated_model import FederatedModel
from utils import load_dataset, split_dataset, remove_outliers
from local_model import LocalModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import multiprocessing
import matplotlib.pyplot as plt

def run_model(model):
    model.fit(StandardScaler(), rules_count=2)
    return model

if __name__ == '__main__':
    number_of_models = 3

    data, feature_names = load_dataset("winequality-red.csv")
    data = remove_outliers(data, neighbors=3)
    print(data.shape)
    data = data[:, :]
    
    # datasets = split_dataset(data, number_of_models)
    datasets = split_dataset(data, number_of_models, overlap=True, block_size=300)
    
    print("Datasets shapes: ")
    shapes = ""
    for d in datasets:
        shapes += str(d.shape)
    print(shapes)
    # exit()
    local_models = list()
    training_times = list()

    for i in range(number_of_models):
        lm = LocalModel(datasets[i], feature_names, epochs=200) 
        local_models.append(lm)

    with multiprocessing.Pool(number_of_models) as p:
        st = time.time()
        local_models = p.map(run_model, [lm for lm in local_models])
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

    fig, axs = plt.subplots(3)
    axs[0].plot([x for x in range(len(local_models[0].error_history))], local_models[0].error_history) 
    axs[0].set_title("Local model 1")
    axs[1].plot([x for x in range(len(local_models[1].error_history))], local_models[1].error_history) 
    axs[1].set_title("Local model 2")
    axs[2].plot([x for x in range(len(local_models[2].error_history))], local_models[2].error_history) 
    axs[2].set_title("Local model 3")
    plt.tight_layout()
    plt.show()    


