from fcm_analyzer import FCMAnalyzer
from random_data_generator import RandomDataGenerator
from tskmodel import TSKModel
from sklearn.metrics import mean_squared_error

import numpy as np
import FuzzySystem as fuzz

rdg = RandomDataGenerator()
fcm_analyzer = FCMAnalyzer(clusters=[x for x in range(2, 5)])

# data = rdg.get_3D_normal_points(size=10, means=(0, 2, 0), stds=(1, 2, 0.5))
data = np.array([[0, 2, 1], [2, 4, 5], [3, 6, 6]])
data_std = np.std(data, axis=1)
print("Data std: ", data_std)
data = data

x1 = fuzz.FuzzySet("X_LOW", fuzz.Gaussmf([data_std[0], 1], universe=[0, 10]))
x2 = fuzz.FuzzySet("X_HIGH", fuzz.Gaussmf([data_std[0], 2], universe=[0, 10]))
x_var = fuzz.FuzzyVariable("X", [x1, x2], universe=[0, 10])
x_var.show()

y1 = fuzz.FuzzySet("Y_LOW", fuzz.Gaussmf([data_std[1], 2], universe=[0, 10]))
y2 = fuzz.FuzzySet("Y_HIGH", fuzz.Gaussmf([data_std[1], 4], universe=[0, 10]))
y_var = fuzz.FuzzyVariable("Y", [y1, y2], universe=[0, 10])
y_var.show()

output1 = fuzz.TSKConsequent(params=[1, 1, 1], function="linear")
output2 = fuzz.TSKConsequent(params=[1, 1, 1], function="linear")

antecedent1 = fuzz.Antecedent(x_var["X_LOW"] & y_var["Y_HIGH"])
antecedent2 = fuzz.Antecedent(x_var["X_HIGH"] & y_var["Y_LOW"])

rule1 = fuzz.FuzzyRule(antecedent1, output1)
rule2 = fuzz.FuzzyRule(antecedent2, output2)

fis = fuzz.FuzzyInferenceSystem([rule1, rule2], and_op="prod", or_op="sum")
inputs = ({"X": 1, "Y": 1})
fis_result = fis.eval(inputs, verbose=True)
result = fuzz.TSKDefuzzifier(fis_result).eval()
print(result)

# clustering_result = fcm_analyzer.fit(data)
# fcm_analyzer.show_fpc()

# tsk_model = TSKModel([rule1, rule2])
# tsk_model.fit(input_data=data[:, :-1], input_labels=["X", "Y"], output_data=data[:, -1])

epochs = 10
lr = 0.01
labels = ["X", "Y"]
error_history = list()

for epoch in range(epochs):
    predicted_outputs = list()

    #  calculating errors
    for input_d, output_d in zip(data[:, :-1], data[:, -1]):
        print(input_d, " ", output_d)
        input_dict = dict()
        for index, label in enumerate(labels):
            input_dict[label] = input_d[index]
        # print(input_dict)
        fis_result = fis.eval(input_dict, verbose=True)
        result = fuzz.TSKDefuzzifier(fis_result).eval()
        predicted_outputs.append(result)
        error = output_d - result
        print("Predicted output: ", result)
        print("Error: ", error)
        print("")
    
    mse = mean_squared_error(data[:, -1], predicted_outputs)
    error_history.append(mse)

    print("Epoch mse: ", mse)
    print("********************************")
    print("\nUPDATE PHASE\n")
    print("********************************")

    # gradient update
    for input_d, output_d in zip(data[:, :-1], data[:, -1]):
        input_dict = dict()

        for index, label in enumerate(labels):
            input_dict[label] = input_d[index]

        fis_result = fis.eval(input_dict, verbose=True)
        print("Firing strength:", fis_result.firing_strength)
        print("Sum of firing strength:", fis_result.firing_strength.sum(axis=0))
        print("Division of firing strength:", fis_result.firing_strength[0] / fis_result.firing_strength.sum(axis=0))
        for rule_index, rule in enumerate(fis.rules):
            cons_params = rule.consequent.get_params()
            
            new_coeffs = list()
            
            for coeff_index in range(len(cons_params)):
                new_coeff = None

                if coeff_index == 0:
                    gradient = lr * mse * fis_result.firing_strength[rule_index] / fis_result.firing_strength.sum(axis=0)
                    new_coeff = cons_params[coeff_index] - gradient
                    print("Gradient: ", gradient)
                    # print("New Coeff: ", new_coeff)
                else:
                    gradient = lr * mse * fis_result.firing_strength[rule_index] / fis_result.firing_strength.sum(axis=0) * input_d[coeff_index - 1]
                    new_coeff = cons_params[coeff_index] - gradient
                    print("Gradient: ", gradient)
                    # print("New Coeff: ", new_coeff)
                # print("")
                new_coeffs.append(new_coeff)

            # print(new_coeffs)
            rule.consequent.set_params(new_coeffs)
            print("")

        for rule in fis.rules:
            print("New params: ", rule.consequent.get_params())

print(error_history)
print("Final error: ", error_history[-1])

for input_d, output_d in zip(data[:, :-1], data[:, -1]):
    input_dict = dict()

    for index, label in enumerate(labels):
        input_dict[label] = input_d[index]

    fis_result = fis.eval(input_dict, verbose=True)
    result = fuzz.TSKDefuzzifier(fis_result).eval()
    print("For input data: ", input_dict, " result is ", result)