
import multiprocessing
import time
import FuzzySystem as fuzz
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler
from agglomerative_clustering import agglomerative_clustering
from tskmodel import TSKModel
from utils import create_fuzzy_variables_from_clusters, create_rules_from_clusters

class FederatedModel:
    def __init__(self, local_models, feature_names, rounds_count=1) -> None:
        self.ffis = None # fuzzy inference system with merged coefficients
        self.local_models = local_models # list of local models
        self.rounds_count = rounds_count # number of federated epochs
        self.feature_names = feature_names # names of features
        self.number_of_fed_rules = 0
        """ 
        final_results is np.array which is a tensor of size:
            (number of local models, number of rounds, number of aggregation functions + 1) 
        and stores errors for each local model for each round for each aggregation (including local fis) 
        """
        lm_count = len(self.local_models)
        self.final_results = np.zeros((lm_count, rounds_count, 4)) 

    @staticmethod
    def run_model(model):
        model.fit()
        return model

    def create_rules(self, rules_count=None) -> list:
        """
        Create federated rules by agglomerative clustering performed on centers and stds of local models fcm result
        """
        scaler = StandardScaler()

        for lm in self.local_models:
            # perform fuzzy c-means clustering - result will be stored as lm object property
            lm.fcm(scaler)

        cluster_centers = list()
        cluster_stds = list()

        for lm in self.local_models:
            fuzzy_partition = None

            if rules_count is None:
                fuzzy_partition = lm.fcm_analyzer.get_best_partition()
            else:
                fuzzy_partition = lm.fcm_analyzer.get_partition(k=rules_count)

            cluster_centers.append(fuzzy_partition['cluster_centers'])
            cluster_stds.append(fuzzy_partition['crisp_cluster_stds'])

        fed_centers, fed_stds = agglomerative_clustering(cluster_centers, cluster_stds, 1)
        fed_centers = np.array(fed_centers)
        fed_stds = np.array(fed_stds)
        
        fuzzy_vars = create_fuzzy_variables_from_clusters(fed_centers, 
                                                    fed_stds, 
                                                    feature_names=self.feature_names, 
                                                    show_fuzzy_vars=False)

        rules = create_rules_from_clusters(fed_centers, fuzzy_vars)
        self.number_of_fed_rules = len(rules)
        
        return rules

    def set_federated_rules_to_local_models(self, fed_rules):
        for lm in self.local_models:
            lm.set_rules(fed_rules)


    def fit(self) -> None:     
        number_of_models = len(self.local_models)

        for model in self.local_models:
            # train test split local datasets
            model.train_test_split()

        for round_index in range(self.rounds_count):
            with multiprocessing.Pool(number_of_models) as p:
                st = time.time()
                self.local_models = p.map(FederatedModel.run_model, self.local_models)
                et = time.time()
                print("Round {} time: {} s".format(round_index+1, et-st))

            i = 0

            for lm in self.local_models:
                print("=" * 200)
                print("Model nr {:2d} \
                    Training error: {:1.6f} \
                    Testing error: {:1.6f}".format(i + 1, lm.error_history[-1], lm.test_mse))
                print("=" * 200)
                i = i + 1

            aggr_rs = self.make_aggregations()
            # print(aggr_rs)
            print("")
            # for key in aggr_rs.keys():
            #     agr_r = aggr_rs[key]
            #     print("Coefficients of {} aggregation".format(key))
            #     for r in agr_r:
            #         print(r.consequent.get_params())
            #     print("")

            # self.merge_local_models()
            self.test_aggregations(aggr_rs, round_index)
        
        print("Final results: ")
        print(self.final_results)

    def arithmetic_mean(self) -> list:
        """
            Returns list of federated rules with coefficients created by arithmetic mean aggregation
        """
        local_rules = list()

        for lm in self.local_models:
            lr = lm.tsk_model.rules
            local_rules.append(lr)

        number_of_rules = len(self.local_models[0].tsk_model.rules)
        print("Number of rules: ", number_of_rules)

        aggregated_coeffs = list()
        for rule_index in range(number_of_rules):
            aggregated_coeffs.append(list())

        for lm in self.local_models:
            for rule_index in range(number_of_rules):
                rule = lm.tsk_model.rules[rule_index]
                for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                    if len(aggregated_coeffs[rule_index]) < coeff_index + 1:
                        aggregated_coeffs[rule_index].append(0)

                    aggregated_coeffs[rule_index][coeff_index] += coeff

        # print("FM rule coeffs: ", federated_coeffs)

        for rule_index in range(len(aggregated_coeffs)):
            rule = aggregated_coeffs[rule_index]
            for coeff_index in range(len(rule)):
                aggregated_coeffs[rule_index][coeff_index] = aggregated_coeffs[rule_index][coeff_index] / len(self.local_models)

        # print("FM rule coeffs after multiply: ", federated_coeffs)
        arithmetic_mean_rules = copy.deepcopy(self.local_models[0].tsk_model.rules)

        for rule, new_coeffs in zip(arithmetic_mean_rules, aggregated_coeffs):
            rule.consequent.set_params(new_coeffs)

        return arithmetic_mean_rules

    def weighted_aritmetic_mean(self) -> list:
        """
            Returns list of federated rules with coefficients created by weighted arithmetic mean aggregation.
            The weights are model_weight parameters from LocalModel objects.
        """
        local_rules = list()

        for lm in self.local_models:
            lr = lm.tsk_model.rules
            local_rules.append(lr)

        number_of_rules = len(self.local_models[0].tsk_model.rules)
        print("Number of rules: ", number_of_rules)

        aggregated_coeffs = list()
        for rule_index in range(number_of_rules):
            aggregated_coeffs.append(list())

        for lm in self.local_models:
            for rule_index in range(number_of_rules):
                rule = lm.tsk_model.rules[rule_index]
                for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                    if len(aggregated_coeffs[rule_index]) < coeff_index + 1:
                        aggregated_coeffs[rule_index].append(0)

                    aggregated_coeffs[rule_index][coeff_index] += coeff * lm.model_weight

        sum_of_weights = 0
        for lm in self.local_models:
            sum_of_weights += lm.model_weight
            print("Local model weight: ", lm.model_weight)

        print("Sum of model weights: ", sum_of_weights)

        for rule_index in range(len(aggregated_coeffs)):
            rule = aggregated_coeffs[rule_index]
            for coeff_index in range(len(rule)):
                aggregated_coeffs[rule_index][coeff_index] = aggregated_coeffs[rule_index][coeff_index] / sum_of_weights

        # print("FM rule coeffs after multiply: ", federated_coeffs)
        weighted_arithmetic_mean_rules = copy.deepcopy(self.local_models[0].tsk_model.rules)

        for rule, new_coeffs in zip(weighted_arithmetic_mean_rules, aggregated_coeffs):
            rule.consequent.set_params(new_coeffs)

        return weighted_arithmetic_mean_rules

    def owa_aggregation(self) -> list:
        """
            Returns list of federated rules with coefficients created by ordered weighted averaging (OWA) aggregation.
            The weights are model_weight parameters from LocalModel objects.
        """
        local_rules = list()

        for lm in self.local_models:
            lr = lm.tsk_model.rules
            local_rules.append(lr)

        # Create OWA weights based on model weights
        owa_weights = list()
        for lm in self.local_models:
            owa_weights.append(lm.model_weight)
            print("Local model weight: ", lm.model_weight)
        
        # print("OWA Weights: ", owa_weights)
        # print("OWA Weights Sum: ", np.sum(owa_weights))
        assert round(np.sum(owa_weights), 3) == 1.0        

        number_of_rules = len(self.local_models[0].tsk_model.rules)
        # print("Number of rules: ", number_of_rules)

        aggregated_coeffs = list()
        for rule_index in range(number_of_rules):
            aggregated_coeffs.append(list())

        for lm in self.local_models:
            for rule_index in range(number_of_rules):
                rule = lm.tsk_model.rules[rule_index]
                for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                    if len(aggregated_coeffs[rule_index]) < coeff_index + 1:
                        aggregated_coeffs[rule_index].append(list())

                    aggregated_coeffs[rule_index][coeff_index].append(coeff)

        # print("This list should be 3-level list with coefficients from local models", aggregated_coeffs)

        # Sort all coefficients lists
        for rule_index in range(len(aggregated_coeffs)):
            rule_coeffs = aggregated_coeffs[rule_index]
            for coeff_index in range(len(rule_coeffs)):
                aggregated_coeffs[rule_index][coeff_index].sort(reverse=True)

        # print("This list should be 3-level SORTED list with coefficients from local models", aggregated_coeffs)

        owa_coeffs = list()
        for rule_index in range(number_of_rules):
            owa_coeffs.append(list())

        for lm in self.local_models:
            for rule_index in range(number_of_rules):
                rule = lm.tsk_model.rules[rule_index]
                for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                    if len(owa_coeffs[rule_index]) < coeff_index + 1:
                        owa_coeffs[rule_index].append(0)

        assert len(owa_coeffs[0]) == len(self.local_models[0].tsk_model.rules[0].consequent.get_params())

        for rule_index in range(len(aggregated_coeffs)):
            rule_coeffs = aggregated_coeffs[rule_index]
            for coeff_index in range(len(rule_coeffs)):
                rule_coeffs_list = aggregated_coeffs[rule_index][coeff_index]
                owa = 0
                
                for single_coeff_index in range(len(rule_coeffs_list)):
                    owa += rule_coeffs_list[single_coeff_index] * owa_weights[single_coeff_index]
                
                owa_coeffs[rule_index][coeff_index] = owa

        print()
        # print("OWA coeffs: ", owa_coeffs)

        owa_rules = copy.deepcopy(self.local_models[0].tsk_model.rules)

        for rule, new_coeffs in zip(owa_rules, owa_coeffs):
            rule.consequent.set_params(new_coeffs)

        return owa_rules

    def make_aggregations(self) -> dict:
        """
        Performs averaging, weighted averaging and OWA aggregation on local models
        """
        aggregated_rules = dict() # dict with pairs: aggregation name, aggregated rules
        aggregated_rules['arithmetic_mean'] = self.arithmetic_mean()
        aggregated_rules['weighted_arithmetic_mean'] = self.weighted_aritmetic_mean()
        aggregated_rules['owa'] = self.owa_aggregation()

        return aggregated_rules

    def merge_local_models(self, aggregations) -> None:
        local_rules = list()

        for lm in self.local_models:
            lr = lm.tsk_model.rules
            local_rules.append(lr)

        number_of_rules = len(self.local_models[0].tsk_model.rules)
        print("Number of rules: ", number_of_rules)

        federated_coeffs = list()
        for rule_index in range(number_of_rules):
            federated_coeffs.append(list())

        for lm in self.local_models:
            for rule_index in range(number_of_rules):
                rule = lm.tsk_model.rules[rule_index]
                for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                    if len(federated_coeffs[rule_index]) < coeff_index + 1:
                        federated_coeffs[rule_index].append(0)

                    federated_coeffs[rule_index][coeff_index] += coeff

        # print("FM rule coeffs: ", federated_coeffs)

        for rule_index in range(len(federated_coeffs)):
            rule = federated_coeffs[rule_index]
            for coeff_index in range(len(rule)):
                federated_coeffs[rule_index][coeff_index] = federated_coeffs[rule_index][coeff_index] / len(self.local_models)

        # print("FM rule coeffs after multiply: ", federated_coeffs)
        federated_rules = self.local_models[0].tsk_model.rules.copy()
        
        for rule, new_coeffs in zip(federated_rules, federated_coeffs):
            rule.consequent.set_params(new_coeffs)

        self.ffis = fuzz.FuzzyInferenceSystem(federated_rules, and_op="prod", or_op="sum")


    def test_aggregations(self, aggregations, round_index) -> None:
        # Store the mse computed on local fis
        lm_index = 0
        for lm in self.local_models:
            self.final_results[lm_index, round_index, 0] = lm.test_mse
            lm_index = lm_index + 1

        # compute mse on local test datasets and decide, whether local or federated
        # fis save to next round
        agrregation_index = 0

        for key in aggregations.keys():
            # aggr is list of rules
            aggr = aggregations[key]
            
            self.ffis = fuzz.FuzzyInferenceSystem(aggr, and_op="prod", or_op="sum")
            print("Aggregation: ", key)
            
            lm_index = 0

            for lm in self.local_models:
                lm.set_fuzzy_inference_system(self.ffis)
                mse = lm.test()
                self.final_results[lm_index, round_index, agrregation_index + 1] = mse
                model_str = ""
                # mse is federated mse, lm.test_mse is local model mse
                if mse > lm.test_mse:
                    model_str = "Local"
                    # lm.restore_local_fis() # if local is better then restore local fis with its own coeffs
                else:
                    model_str = "Federated " + key
                lm.restore_local_fis()
                print("=" * 100)
                print("Dataset nr {:2d} \
                    Testing error: {:1.6f} \
                    Better model: {:12s}".format(lm_index + 1, mse, model_str))
                print("=" * 100)
                lm_index = lm_index + 1
            
            agrregation_index = agrregation_index + 1
        
        # which aggregation is best
        min_mse_indices = np.argmin(self.final_results, axis=2)
        
        for r in range(min_mse_indices.shape[0]):
            if min_mse_indices[r, 0] == 0:
                print("Local")
            elif min_mse_indices[r, 0] == 1:
                print("Mean")
            elif min_mse_indices[r, 0] == 2:
                print("Weighted Mean")
            elif min_mse_indices[r, 0] == 3:
                print("OWA")
                
                