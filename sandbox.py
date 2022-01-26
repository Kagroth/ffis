import numpy as np
import FuzzySystem as fs

#INPUTS
x1 = fs.FuzzySet('X_1', fs.Trapmf([-1.5, -1.5, -1, 1]))
x2 = fs.FuzzySet('X_2', fs.Trapmf([-1, 1, 1.5, 1.5]))
mf1 = fs.FuzzyVariable('X', [x1, x2], universe=[-1.5, 1.5])
mf1.show()

y1 = fs.FuzzySet('Y_1', fs.Trapmf([-1.5, -1.5, -0.5, 1.2]))
y2 = fs.FuzzySet('Y_2', fs.Trapmf([-1, 1, 1.5, 1.5]))
mf2 = fs.FuzzyVariable('Y', [y1, y2], universe=[-1.5, 1.5])
# mf2.show()

z1 = fs.FuzzySet('Z_1', fs.Trapmf([-1.5, -1, 0.3, 1.2]))
z2 = fs.FuzzySet('Z_2', fs.Trapmf([-0.4, 1, 1.3, 1.5]))
mf3 = fs.FuzzyVariable('Z', [z1, z2], universe=[-1.5, 1.5])
# mf3.show()

#OUTPUT
# def f1(x1, x2, x3, a1, a2, a3, a4): return x1 * a1 + x2 * a2 + x3 * a3 + a4
def f1(x1, x2, x3, a0, a1, a2, a3): return a0 + x1 * a1 + x2 * a2 + x3 * a3
def f2(x1, x2, x3, a1, a2, a3, a4): return x1 * a1 + x2 * a2 + x3 * a3 + a4
def f3(x1, x2, x3, a1, a2, a3, a4): return x1 * a1 + x2 * a2 + x3 * a3 + a4

output = fs.TSKConsequent(params=np.array([2, 2.5, 1, 0]), function=f1)
# output1 = fs.TSKConsequent(params=np.array([4, 4.5, 1, 0]), function=f2)
# output2 = fs.TSKConsequent(params=np.array([1, 2, 1, 0]), function=f3)
output1 = fs.TSKConsequent(params=np.array([2, 2.5, 1, 0]), function="linear")
# output1 = fs.TSKConsequent(params=np.array([0, 4.5, 1, 4]), function="linear")
# output2 = fs.TSKConsequent(params=np.array([0, 2, 1, 1]), function="linear")

ant1  = fs.Antecedent(mf1['X_1'] & mf2['Y_2'] & mf3['Z_1'])
rule1 = fs.FuzzyRule(ant1, output)

ant2  = fs.Antecedent(mf1['X_2'] & mf2['Y_1'] & mf3['Z_2'])
rule2 = fs.FuzzyRule(ant2, output)

ant3  = fs.Antecedent(mf1['X_1'] & mf2['Y_2'] & mf3['Z_1'])
rule3 = fs.FuzzyRule(ant3, output1)

ant4  = fs.Antecedent(mf1['X_2'] & mf2['Y_1'] & mf3['Z_2'])
rule4 = fs.FuzzyRule(ant4, output1)

#Building the FIS
fis = fs.FuzzyInferenceSystem([rule1, rule2], and_op='prod', or_op='sum')
fis2 = fs.FuzzyInferenceSystem([rule3, rule4], and_op='prod', or_op='sum')

#FIS Evaluation
inputs = ({'X':-1.5, 'Y': 1, 'Z': -0.3})
result = fis.eval(inputs, verbose=True)
result2 = fis2.eval(inputs, verbose=True)
r = fs.TSKDefuzzifier(result).eval()
r2 = fs.TSKDefuzzifier(result2).eval()
print(r)
print(r2)
# inputs = ({'X': [-1.5, -4], 'Y': [2, -1]})
# result = fis.eval(inputs, verbose=True)
# fs.TSKDefuzzifier(result).eval()