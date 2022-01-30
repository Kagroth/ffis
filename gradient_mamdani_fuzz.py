import enum
import random
import numpy as np
import FuzzySystem as fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = np.array([[0, 2, 1], [2, 4, 5], [3, 6, 6]])

# X variable
x_low = fuzz.FuzzySet("X_LOW", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[0])], universe=[-10, 10]))
x_high = fuzz.FuzzySet("X_HIGH", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[0])], universe=[-10, 10]))
x_var = fuzz.FuzzyVariable("X", [x_low, x_high], universe=[-10, 10])
# x_var.show()

# Y variable
y_low = fuzz.FuzzySet("Y_LOW", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[1])], universe=[-10, 10]))
y_high = fuzz.FuzzySet("Y_HIGH", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[1])], universe=[-10, 10]))
y_var = fuzz.FuzzyVariable("Y", [y_low, y_high], universe=[-10, 10])
# y_var.show()

# Z - output
z_low = fuzz.FuzzySet("Z_LOW", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[2])], universe=[-10, 10]))
z_high = fuzz.FuzzySet("Z_HIGH", fuzz.Gaussmf([1, random.uniform(0, data.mean(axis=0)[2])], universe=[-10, 10]))
z_var = fuzz.FuzzyVariable("Z", [z_low, z_high], universe=[-10, 10])
# z_var.show()

ant1 = fuzz.Antecedent(x_var['X_LOW'] & y_var['Y_LOW'])
cons1 = fuzz.Consequent(z_var['Z_LOW'])
rule1 = fuzz.FuzzyRule(ant1, cons1)

ant2 = fuzz.Antecedent(x_var['X_HIGH'] & y_var['Y_HIGH'])
cons2 = fuzz.Consequent(z_var['Z_HIGH'])
rule2 = fuzz.FuzzyRule(ant2, cons2)

fis = fuzz.FuzzyInferenceSystem([rule1, rule2], and_op='prod', or_op='sum')
input_d = {"X": 3, "Y": 6}
# fis_result = fis.eval((input_d), verbose=True)
# print(fis_result)
# print([output[0][1].firing_strength for output in fis_result._outputs])
# print(sum([output[0][1].firing_strength for output in fis_result._outputs]))
# result = fuzz.Centroid(fis_result).eval()
# print(result['Z'])


lr = 0.2

# error
# err = result['Z'] - 6
# print("error: ", err)
error_history = list()

for epoch in range(500):
    mse = np.Inf

    for d in data:
        input_d = {"X": d[0], "Y": d[1]}
        # print(input_d)
        fis_result = fis.eval((input_d), verbose=False)
        result = fuzz.Centroid(fis_result).eval()
        err = result['Z'] - d[2]
        # print("error: ", err)

        # Learning step
        for rule_index, rule in enumerate(fis.rules):
            # Antecedent membership functions update
            # print(fis_result.firing_strength)
            rule_firing_strength = fis_result._outputs[rule_index][0][1].firing_strength # firing strength of this rule
            sum_of_firing_strength = sum([output[0][1].firing_strength for output in fis_result._outputs])
            premise_variables = rule.antecedent.propositions.get_tuples()
            consequent_variable = rule.consequent
            consequent_var_name, consequent_fuzzy_set_name = consequent_variable.propositions[0].get_tuple() # propositions[0] because there is only one output variable
            consequent_fuzzy_set = consequent_variable.fuzzy_variables[consequent_var_name].get(consequent_fuzzy_set_name)

            b = consequent_fuzzy_set.mf.c

            # input fuzzy vars update center and spread
            for var_name, fuzzy_set_name in premise_variables:
                # print(var_name, fuzzy_set_name)
                fuzzy_var = rule.antecedent.fuzzy_variables[var_name]
                fuzzy_set = fuzzy_var.get(fuzzy_set_name)

                center = fuzzy_set.mf.c
                spread = fuzzy_set.mf.spread
                inp = input_d[var_name]
                
                common_grad = err * ((b - result['Z']) / sum_of_firing_strength) * rule_firing_strength
                center_grad = lr * common_grad * (inp - center) / (spread ** 2)
                spread_grad = lr * common_grad * ((inp - center) ** 2) / (spread ** 3)

                new_center = center - center_grad
                new_spread = spread - spread_grad

                fuzzy_set.mf = fuzz.Gaussmf([new_spread, new_center], universe=[-10, 10])
                # print(fuzzy_var.get(fuzzy_set_name))
            
            # Consequent membership function update
            consequent_center_grad = lr * err * (rule_firing_strength / sum_of_firing_strength)
            consequent_new_center = consequent_fuzzy_set.mf.c - consequent_center_grad
            consequent_fuzzy_set.mf = fuzz.Gaussmf([consequent_fuzzy_set.mf.spread, consequent_new_center], universe=[-10, 10])
            # print(consequent_variable.fuzzy_variables[consequent_var_name].get(consequent_fuzzy_set_name))

    predicted_outputs = list()

    for d in data:
        input_d = {"X": d[0], "Y": d[1]}
        fis_result = fis.eval((input_d), verbose=False)
        result = fuzz.Centroid(fis_result).eval()
        predicted_outputs.append(result['Z'])

    # print(data[:, -1])
    # print(predicted_outputs)
    mse = mean_squared_error(data[:, -1], predicted_outputs)
    error_history.append(mse)
    print("Epoka {}, blad: {}".format(epoch + 1, mse))

    if mse < 0.005:
        print("Koniec uczenia")
        print("Liczba epok: ", epoch + 1)
        print("Error: ", error_history[-1])
        break

print(fuzz.Centroid(fis.eval(({"X": 3, "Y": 6}))).eval()['Z'])

print(data.mean(axis=0))
plt.plot(error_history)
plt.show()

for var in fis.rules[0].antecedent.fuzzy_variables:
    fis.rules[0].antecedent.fuzzy_variables[var].show()

fis.rules[0].consequent.fuzzy_variables['Z'].show()

# # input mf center updates
# # X_LOW
# x_low_center = x_low.mf.c
# x_low_spread = x_low.mf.spread
# b = z_low.mf.c
# rule_fire_strength = fis_result._outputs[0][0][1].firing_strength
# x_input = 3

# common_grad = err * ((b - result['Z']) / sum([output[0][1].firing_strength for output in fis_result._outputs]))
# common_grad = common_grad * rule_fire_strength

# x_low_center_grad = lr * common_grad * ((x_input - x_low_center) / (x_low_spread ** 2))
# x_low_new_center = x_low_center - x_low_center_grad

# print("X_LOW center gradient: ", x_low_center_grad)
# print("X_LOW new center: ", x_low_new_center)

# # input mf spread updates
# # X_LOW
# x_low_spread = x_low.mf.spread
# x_low_spread_grad = lr * common_grad * ((x_input - x_low_center) ** 2) / (x_low_spread ** 3) 
# x_low_new_spread = x_low_spread - x_low_spread_grad
# print("X_LOW spread gradient: ", x_low_spread_grad)
# print("X_LOW new spread: ", x_low_new_spread)

# x_low.mf = fuzz.Gaussmf([x_low_new_spread, x_low_new_center], universe=[-10, 10])

# # output mf updates
# z_low_center = z_low.mf.c
# rule_fire_strength = fis_result._outputs[0][0][1].firing_strength
# z_low_center_grad = lr * err * (rule_fire_strength / sum([output[0][1].firing_strength for output in fis_result._outputs]))
# z_low_new_center = z_low_center - z_low_center_grad
# print("New center for z_low mf: ", z_low_new_center)
# z_low.mf = fuzz.Gaussmf([1, z_low_new_center], universe=[-10, 10])

# z_high_center = z_high.mf.c
# rule_fire_strength = fis_result._outputs[1][0][1].firing_strength
# print(rule_fire_strength)
# z_high_center_grad = lr * err * (rule_fire_strength / sum([output[0][1].firing_strength for output in fis_result._outputs]))
# z_high_new_center = z_high_center - z_high_center_grad
# print("New center for z_high mf: ", z_high_new_center)
# z_high.mf = fuzz.Gaussmf([1, z_high_new_center], universe=[-10, 10])

# print(fis.rules[0])
# print(fis.rules[0].antecedent)
# print(type(fis.rules[0].antecedent))
# print(fis.rules[0].antecedent.get("X"))
# print(fis.rules[0].antecedent.fuzzy_variables['X'])
# print(type(fis.rules[0].antecedent.propositions))
# print(fis.rules[0].antecedent.propositions.fuzzy_variables['X'])
# print(fis.rules[0].antecedent.propositions.get_tuples())
# print(fis.rules[1].antecedent.propositions.get_tuples())
