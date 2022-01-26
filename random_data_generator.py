import numpy as np

class RandomDataGenerator:
    def __init__(self) -> None:
        pass

    def get_uniform_points(self, size=1, lower_bound=0, upper_bound=10) -> np.ndarray:
        points = np.random.uniform(low=lower_bound, high=upper_bound, size=size)

        return points

    def get_standard_normal_points(self, size=1) -> np.ndarray:
        points = np.random.standard_normal(size=size)

        return points
    
    def get_normal_points(self, size=1, mean=1, std=0) -> np.ndarray:
        points = np.random.normal(mean, std, size)

        return points

    def get_2D_uniform_points(self, size=1, x_range=(0, 10), y_range=(0, 10)) -> np.ndarray:
        """
            Returns 2D column vectors generated randomly from uniform distribution. 
        """
        x_feature = self.get_uniform_points(size=size, lower_bound=x_range[0], upper_bound=x_range[1])
        y_feature = self.get_uniform_points(size=size, lower_bound=y_range[0], upper_bound=y_range[1])
        all_data = np.column_stack((x_feature, y_feature))
        return all_data.T
    
    def get_2D_standard_normal_points(self, size=1) -> np.ndarray:
        """
            Returns 2D column vectors generated randomly from standard normal distribution. 
        """
        x_feature = np.random.standard_normal(size=size)
        y_feature = np.random.standard_normal(size=size)
        all_data = np.vstack((x_feature, y_feature))
        return all_data
    
    def get_2D_normal_points(self, size=1, mean=1, shift_x=0, shift_y=0) -> np.ndarray:
        """
            Returns 2D column vectors generated randomly from normal distribution. 
        """
        # to chyba powinno byc na odwrot (mean i shift)
        x_feature = np.random.standard_normal(size) * mean + shift_x
        y_feature = np.random.standard_normal(size) * mean + shift_y
        all_data = np.vstack((x_feature, y_feature))
        return all_data
    
    def get_3D_uniform_points(self, size, value_range) -> np.ndarray:
        vectors_2d = self.get_2D_uniform_points(size, value_range, value_range)
        z_feature = self.get_uniform_points(size, lower_bound=value_range[0], upper_bound=value_range[1])
        all_data = np.column_stack((vectors_2d.T, z_feature))
        return all_data.T

    def get_3D_normal_points(self, size=1, means=(0, 0, 0), stds=(1, 1, 1)) -> np.ndarray:
        """
            Returns 3D column vectors generated randomly from normal distribution
        """
        x = self.get_normal_points(size, means[0], stds[0])
        y = self.get_normal_points(size, means[1], stds[1])
        z = self.get_normal_points(size, means[2], stds[2])
        
        all_data = np.vstack((x, y, z))
        return all_data