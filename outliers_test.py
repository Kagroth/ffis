from sklearn import datasets
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import load_dataset
import numpy as np

data, feature_names = load_dataset("winequality-red.csv")

# print("Before Forest: ", data.shape)
# lof = LocalOutlierFactor(n_neighbors=3)
# yhat = lof.fit_predict(data)
# mask = yhat != -1
# filtered = data[mask, :]
# print("After Forest: ", filtered.shape)

indices = np.random.choice(data.shape[0], 10)
print(indices)
print(data[indices, :])
print(np.array_split(data, 2))
print(type(np.array_split(data, 2)))