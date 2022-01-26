import enum
import math
import numpy as np
import FuzzySystem as fuzz

class TSKModel:
    def __init__(self, rules: list, epochs: int=10, lr: float=0.1, eg: float=0.1) -> None:
        self.rules = rules
        self.input_data = None
        self.output_data = None
        self.epochs = epochs
        self.lr = lr
        self.eg = eg
        self.fis = None

    def fit(self, input_data: np.ndarray, input_labels: list, output_data: np.ndarray) -> list:
        self.input_data = input_data
        self.input_labels = input_labels
        self.output_data = output_data

        fis = fuzz.FuzzyInferenceSystem(self.rules, and_op="prod", or_op="sum")
        error_history = list()
        rules = self.rules

        for epoch in range(self.epochs):
            epoch_error = 0
            
            # init empty lists to store new coefficients. Number of lists = number of rules
            epoch_coeffs = list()
            for rule_index in range(len(rules)):
                epoch_coeffs.append(list())

            for input_vector, output_value in zip(self.input_data, self.output_data):
                
                input_dict = dict()
                for index, label in enumerate(self.input_labels):
                    input_dict[label] = input_vector[index]
                # create input for tsk fis from input_vector
                # inputs = ({"X": 3, "Y": 2})
                inputs = (input_dict)
                fis_result = fis.eval(inputs, verbose=False)
                result = fuzz.TSKDefuzzifier(fis_result).eval()
                # error = output_value - result
                error = ((output_value - result) ** 2) / 2
                new_coeffs = self.coefficients_update(input_vector, fis_result, fis.rules, error)

                for rule_index, rule in enumerate(rules):
                    nc = new_coeffs[rule_index]
                    epoch_coeffs[rule_index].append(nc)
                
                epoch_error += error

            # print("Epoch error sum: ", epoch_error)
            epoch_error = epoch_error / len(self.input_data)
            # print("Epoch error mean: ", epoch_error)
            # new coefficients are mean of updated coefficients computed for every data point pair
            new_coeffs = self.coefficients_mean(epoch_coeffs) 
            
            # update rules - set new coefficients
            rules = self.update_rules(rules, new_coeffs)
            fis = fuzz.FuzzyInferenceSystem(rules, and_op="prod", or_op="sum")

            error_history.append(epoch_error)

            if epoch_error < self.eg:
                # print("Epoch {}, error: {}".format(epoch + 1, epoch_error))
                self.fis = fis
                return error_history
            
            # print("Epoch {}, error: {}".format(epoch + 1, epoch_error))
            epoch += 1

        print("End of training")
        self.fis = fis
        
        return error_history

    def coefficients_update(self, input_vector, fis_result, rules, error) -> list:
        """
        Compute gradients and new coefficients for current FIS result (results for current data point) 
        """
        sum_of_firing_strength = fis_result.firing_strength.sum(axis=0)
        new_coeffs = list()

        for rule_index, rule in enumerate(rules):
            rule_firing_strength = fis_result.firing_strength[rule_index]

            rule_new_coeffs = list()    

            for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                new_coeff = None

                if coeff_index == 0:
                    # first coefficient without 
                    new_coeff = coeff - self.lr * error * (rule_firing_strength / sum_of_firing_strength)
                else:
                    # other coefficients
                    new_coeff = coeff - self.lr * error * (rule_firing_strength / sum_of_firing_strength) * input_vector[coeff_index - 1]
                
                rule_new_coeffs.append(new_coeff)
            
            new_coeffs.append(rule_new_coeffs)

        return new_coeffs

    def coefficients_mean(self, epoch_coeffs) -> list:
        """
        Compute new coefficients as mean of coefficients computed for every data pair in one epoch
        """
        final_coeffs = list()

        for rule_index in range(len(epoch_coeffs)):
            rule_new_coeffs = epoch_coeffs[rule_index]

            rule_coeff_sums = list()
            
            # Number of coefficients equals number of rules (epoch_coeffs length) + 1
            for _ in range(len(epoch_coeffs[0][0])):
                rule_coeff_sums.append(0)
            
            for new_coeffs in rule_new_coeffs:
                for coeff_index in range(len(new_coeffs)):
                    rule_coeff_sums[coeff_index] += new_coeffs[coeff_index]
            
            rule_coeff_means = list()

            for coeff_sum in rule_coeff_sums:
                rule_coeff_means.append(coeff_sum / len(epoch_coeffs[0])) # divide sum by count of data points, for every rule the number of data points is equal

            final_coeffs.append(rule_coeff_means)    

        return final_coeffs

    def update_rules(self, rules, new_coefficients) -> list:
        """
        Set new coefficients for every rule
        """
        for rule, new_coeff in zip(rules, new_coefficients):
            rule.consequent.set_params(new_coeff)
        
        return rules

    def compute(self, input_dict: dict) -> float:
        fis_result = self.fis.eval(input_dict)
        result = fuzz.TSKDefuzzifier(fis_result).eval()
        
        return result