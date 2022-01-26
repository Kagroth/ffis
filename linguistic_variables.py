
from skfuzzy.membership import gaussmf

def membership_function_from_cluster(cluster: object, feature: int=0):
    x = list(range(-10, 10, 1))
    return gaussmf(x, cluster[feature], 1)