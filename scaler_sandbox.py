from sklearn.preprocessing import MinMaxScaler
from random_data_generator import RandomDataGenerator
import numpy as np

rdg = RandomDataGenerator()
points1 = rdg.get_3D_normal_points(size=5, means=(0, 0, 0), stds=(1, 1, 1))
points2 = rdg.get_3D_normal_points(size=5, means=(10, 0, 0), stds=(1, 1, 1))
points3 = rdg.get_3D_normal_points(size=5, means=(0, 10, 0), stds=(1, 1, 1))
points4 = rdg.get_3D_normal_points(size=5, means=(0, 0, 10), stds=(1, 1, 1))
points5 = rdg.get_3D_normal_points(size=5, means=(10, 10, 0), stds=(1, 1, 1))
data = np.hstack((points1, points2, points3, points4, points5))

print(data.T)

scaler = MinMaxScaler()
print(scaler.fit(data.T))
print(scaler.data_min_)
print(scaler.data_max_)
print(scaler.transform(data.T))