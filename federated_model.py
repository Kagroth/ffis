

from tskmodel import TSKModel
import FuzzySystem as fuzz

class FederatedModel:
    def __init__(self) -> None:
        self.ffis = None # fuzzy inference system with merged coefficients

    def merge_local_models(self, local_models) -> None:
        self.local_models = local_models
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


    def test(self) -> None:
        # compute mse on local test datasets
        pass