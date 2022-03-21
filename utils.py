
import numpy as np
import FuzzySystem as fuzz
import random
import os
from sklearn.neighbors import LocalOutlierFactor

def load_dataset(filename) -> tuple:
    """
    Returns tuple[np.ndarray, list] with dataset and feature names
    """
    dataset_path = os.path.join(os.getcwd(), "datasets", filename)
    feature_names = list()
    with open(dataset_path, "r") as f:
        headers = f.readline()
        headers_list = headers.split(";")
        headers_list = [header.replace("\"", "").replace("\n", "") for header in headers_list]
        feature_names = headers_list

    data = np.genfromtxt(dataset_path, delimiter=";", skip_header=True)
    
    return data, feature_names

def feature_std(dataset: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation of each feature of dataset
    
    Parameters:
    - dataset: ndarray of shape RxC, where R is number of vectors and C is the number of features
    
    Return: ndarray of shape 1xC which is standard deviation of each feature
    """
    std_arr = np.std(dataset, axis=0)

    return std_arr

def create_fuzzy_variable(cluster_centers: np.ndarray, feature_name: str, cluster_stds: np.float64) -> fuzz.FuzzyVariable:
    """
    To do:
    - include standard deviation of feature in crisp clusters (currently std = 1)
    - include feature universe range (currently from -15 to 15)

    Create fuzzy variable for feature from cluster centers

    Parameters:
    - cluster_centers: Feature coord of fuzzy c-means clusters which gonna be used
    as centers in Gaussian membership functions creation 
    - feature_name: feature label e.g. X, Y, acidity, speed etc
    
    Return: FuzzyVariable of feature described by cluster_centers
    """
    clusters_data = np.vstack((cluster_centers, cluster_stds)).T
    clusters_data = clusters_data[clusters_data[:, 0].argsort()]

    FEATURE_NAME_SEPARATOR = "_"
    feature_center_name = {}
    fuzzy_sets = []

    for index in range(clusters_data.shape[0]):
        feature_full_name = feature_name + FEATURE_NAME_SEPARATOR + str(index)
        feature_center_name[feature_full_name] = clusters_data[index][0]
        fuzzy_set = fuzz.FuzzySet(feature_full_name, fuzz.Gaussmf([clusters_data[index][1], clusters_data[index][0]], universe=[-2, 2]))
        fuzzy_sets.append(fuzzy_set)
    
    fuzzy_var = fuzz.FuzzyVariable(feature_name, fuzzy_sets, universe=[-2, 2])
    
    return fuzzy_var

def create_fuzzy_variables_from_clusters(cluster_centers: np.ndarray, cluster_stds: np.ndarray, feature_names: list, show_fuzzy_vars: bool = False) -> list:
    """
    Creates Gaussian membership fuzzy variables for given fuzzy cluster centers. 
    Fuzzy variable is a collection of fuzzy sets called universe.

    Parameters:
    - cluster_centers: Center of clusters created by fuzzy cmeans.
    - feature_names: Names of the features of dataset

    Return: list of fuzzy variables
    """
    assert cluster_centers.shape[1] == len(feature_names)
    features = list()

    for index in range(0, cluster_centers.shape[1] - 1): # for every feature (column) except the last column (the output)
        feature = create_fuzzy_variable(cluster_centers[:, index], feature_names[index], cluster_stds[:, index])
        features.append(feature)

        if show_fuzzy_vars:
            feature.show()
    
    return features

def create_rules_from_clusters(cluster_centers: np.ndarray, fuzzy_variables: list) -> None:
    """    
    Create rules from clusters. For every cluster center and for every cluster feature
    find the fuzzy set which center equals cluster feature and add that fuzzy set to
    antecedent of rule, connected by 'and' operator

    Parameters:
    - cluster_centers
    - fuzzy_variables

    Return list of FuzzyRule objects 
    """
    rules = list()

    for center in cluster_centers:
        # center.shape is a tuple of length 1 - e.g (2,)
        feature_count = center.shape[0] - 1 # number of input features, the last feature is output
        antecedent_premise = None           # antecedent_premise is an fuzzy premise formulated created by evaluating '&' operator on fuzzy sets 
        ant = None                          # ant is an Antecedent type object

        for index in range(feature_count):
            fv = fuzzy_variables[index]
            feature_center = center[index]

            for fs in fv.fuzzysets:
                fset_center = fs.mf.params[1] # get center of Gaussian mf
                
                if fset_center == feature_center:
                    if antecedent_premise is None:
                        antecedent_premise = fv[fs.name]
                    else:
                        antecedent_premise = antecedent_premise & fv[fs.name]
                    break
            
            ant = fuzz.Antecedent(antecedent_premise)
        
        consequent_params = [random.uniform(-1, 1) for _ in range(feature_count + 1)] # number of coefficients is equal to input features + 1
        output = fuzz.TSKConsequent(params=consequent_params, function="linear")
        rule = fuzz.FuzzyRule(ant, output)
        rules.append(rule)
    
    return rules


def split_dataset(dataset, blocks_count, overlap=False, block_size=None) -> list:
    """
    Splits dataset into blocks_count datasets
    
    Parameters
    - dataset
    - blocks_count

    Returns list of datasets
    """
    if overlap and dataset.shape[0] > block_size:
        datasets = list()
        for i in range(blocks_count):
            indices = np.random.choice(dataset.shape[0], block_size)
            datasets.append(dataset[indices, :])
        
        return datasets

    np.random.shuffle(dataset)
    return np.array_split(dataset, blocks_count)



def remove_outliers(dataset, neighbors=20) -> np.ndarray:
    lof = LocalOutlierFactor(n_neighbors=neighbors) # 20 is default value
    yhat = lof.fit_predict(dataset)
    mask = yhat != -1
    filtered_data = dataset[mask, :]
    
    return filtered_data