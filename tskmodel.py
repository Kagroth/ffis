import numpy as np
import FuzzySystem as fuzz
from sklearn.metrics import mean_squared_error

class TSKModel:
    def __init__(self, rules: list, epochs: int=10, lr: float=0.1, momentum: float=0.9, eg: float=0.1) -> None:
        self.rules = rules
        self.input_data = None
        self.output_data = None
        self.epochs = epochs
        self.lr = lr
        self.eg = eg
        self.fis = None
        self.prev_coeffs = None
        self.momentum = momentum

    def fit(self, input_data: np.ndarray, input_labels: list, output_data: np.ndarray) -> list:
        self.input_data = input_data
        self.input_labels = input_labels
        self.output_data = output_data

        self.prev_coeffs = list()

        for rule_index, rule in enumerate(self.rules):
            self.prev_coeffs.append(list())
            self.prev_coeffs[rule_index] = rule.consequent.get_params()

            # for coeff_index in range(len(rule.consequent.get_params())):
                # self.prev_coeffs[rule_index].append(0)
            
            # print("Init coeffs of rule {}: {}".format(rule_index+1, self.prev_coeffs[rule_index]))
            
        fis = fuzz.FuzzyInferenceSystem(self.rules, and_op="prod", or_op="sum")
        error_history = list()
        rules = self.rules

        for epoch in range(self.epochs):
            epoch_error = 0
            
            # init empty lists to store new coefficients. Number of lists = number of rules
            epoch_coeffs = list()
            for rule_index in range(len(rules)):
                epoch_coeffs.append(list())

            predicted_outputs = list()

            for input_vector, output_value in zip(self.input_data, self.output_data):
                
                input_dict = dict()
                for index, label in enumerate(self.input_labels):
                    input_dict[label] = input_vector[index]
                
                inputs = (input_dict) # create input for tsk fis from input_vector
                fis_result = fis.eval(inputs, verbose=False)
                result = fuzz.TSKDefuzzifier(fis_result).eval()
                error = result - output_value
                predicted_outputs.append(result)
                new_coeffs = self.coefficients_update(input_vector, fis_result, fis.rules, error)

                for rule_index, rule in enumerate(rules):
                    nc = new_coeffs[rule_index]
                    epoch_coeffs[rule_index].append(nc)
                
            # new coefficients are mean of updated coefficients computed for every data point pair
            epoch_error = mean_squared_error(predicted_outputs, self.output_data)

            new_coeffs = self.coefficients_mean(epoch_coeffs) 
            
            # update rules - set new coefficients
            rules = self.update_rules(rules, new_coeffs)
            fis = fuzz.FuzzyInferenceSystem(rules, and_op="prod", or_op="sum")

            error_history.append(epoch_error)

            if epoch_error < self.eg:
                # print("Epoch {}, error: {}".format(epoch + 1, epoch_error))
                self.fis = fis
                return error_history
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print("Epoch {}, error: {}".format(epoch + 1, epoch_error))
            
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

            grad = self.lr * error * (rule_firing_strength / sum_of_firing_strength)

            for coeff_index, coeff in enumerate(rule.consequent.get_params()):
                new_coeff = None

                if coeff_index == 0:
                    # first coefficient without input value
                    new_coeff = coeff - grad + self.momentum * (coeff - self.prev_coeffs[rule_index][coeff_index])
                else:
                    # other coefficients
                    new_coeff = coeff - grad * input_vector[coeff_index - 1] + self.momentum * (coeff - self.prev_coeffs[rule_index][coeff_index])
                
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
        for rule_index, rule in enumerate(rules):
            self.prev_coeffs[rule_index] = rule.consequent.get_params()

        for rule, new_coeff in zip(rules, new_coefficients):
            rule.consequent.set_params(new_coeff)
        
        return rules

    def compute(self, input_dict: dict) -> float:
        fis_result = self.fis.eval(input_dict)
        result = fuzz.TSKDefuzzifier(fis_result).eval()
        
        return result
    
    def test(self, input_data, output_data) -> float:
        results = list()

        for input_vector in input_data:
            input_dict = dict()

            for index, label in enumerate(self.input_labels):
                input_dict[label] = input_vector[index]
            
            result = self.compute(input_dict)
            results.append(result)
        
        return mean_squared_error(output_data, results)