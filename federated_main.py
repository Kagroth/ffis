from numpy import number
from federated_model import FederatedModel
from utils import load_dataset, split_dataset, remove_outliers
from local_model import LocalModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

number_of_models = 6

data, feature_names = load_dataset("winequality-red.csv")
data = remove_outliers(data, neighbors=3)
data = data[:240, :]
datasets = split_dataset(data, number_of_models)

print("Datasets shapes: ")
shapes = ""
for d in datasets:
    shapes += str(d.shape)
print(shapes)

local_models = list()
training_times = list()

for i in range(number_of_models):
    lm = LocalModel(datasets[i], feature_names, epochs=100) 
    local_models.append(lm)

for lm in local_models:
    st = time.time()
    lm.fit(MinMaxScaler())
    et = time.time()
    training_times.append(et - st)

i = 0

for lm in local_models:
    print("=" * 200)
    print("Model nr {:2d} \
        Training time: {:4.5f} \
        Training error: {:1.6f} \
        Testing error: {:1.6f}".format(i + 1, training_times[i], lm.error_history[-1], lm.test_mse))
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
    print("=" * 100)
    print("Dataset nr {:2d} \
        Testing error: {:1.6f}".format(i + 1, mse))
    print("=" * 100)
    i = i + 1