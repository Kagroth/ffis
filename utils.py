
import numpy as np
import FuzzySystem as fuzz
import random

def feature_std(dataset: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation of each feature of dataset
    
    Parameters:
    - dataset: ndarray of shape RxC, where R is number of vectors and C is the number of features
    
    Return: ndarray of shape 1xC which is standard deviation of each feature
    """
    std_arr = np.std(dataset, axis=0)

    return std_arr

def create_fuzzy_variable(cluster_centers: np.ndarray, feature_name: str) -> fuzz.FuzzyVariable:
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
    centers = cluster_centers
    centers_sorted = np.sort(centers)

    FEATURE_NAME_SEPARATOR = "_"
    feature_center_name = {}
    fuzzy_sets = []

    for index in range(len(centers_sorted)):
        feature_full_name = feature_name + FEATURE_NAME_SEPARATOR + str(index)
        feature_center_name[feature_full_name] = centers_sorted[index]
        fuzzy_set = fuzz.FuzzySet(feature_full_name, fuzz.Gaussmf([1, centers_sorted[index]], universe=[-10, 10]))
        fuzzy_sets.append(fuzzy_set)
    
    fuzzy_var = fuzz.FuzzyVariable(feature_name, fuzzy_sets, universe=[-10, 10])
    
    return fuzzy_var

def create_fuzzy_variables_from_clusters(cluster_centers: np.ndarray, feature_names: list, show_fuzzy_vars: bool = False) -> list:
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

    for index in range(0, cluster_centers.shape[1]): # for every feature (column)
        feature = create_fuzzy_variable(cluster_centers[:, index], feature_names[index])
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
        
        consequent_params = [random.uniform(0, 1) for _ in range(feature_count + 1)] # number of coefficients is equal to input features + 1
        output = fuzz.TSKConsequent(params=consequent_params, function="linear")
        # output = fuzz.TSKConsequent(params=np.ndarray([1, 1, 1]), function="linear")
        # output = fuzz.TSKConsequent(function="linear")
        # output = fuzz.TSKConsequent(params=4, function="constant")
        rule = fuzz.FuzzyRule(ant, output)
        rules.append(rule)
    
    return rules