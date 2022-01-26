import FuzzySystem as fs
import numpy as np

#INPUTS
print('*** Inputs ***')
service_poor = fs.FuzzySet('poor', fs.Gaussmf([1.5, 0]))
service_good = fs.FuzzySet('good', fs.Gaussmf([1.5, 5]))
service_excellent = fs.FuzzySet('excellent', fs.Gaussmf([1.5, 10]))
service = fs.FuzzyVariable('service',[service_poor, service_good, service_excellent], universe=[0, 10])
print('Service')
service.show()

food_rancid = fs.FuzzySet('rancid', fs.Trapmf([0,0,1,3]))
food_delicious = fs.FuzzySet('delicious', fs.Trapmf([7,9,10,10]))
food = fs.FuzzyVariable('food', [food_rancid, food_delicious], universe=[0, 10])
print('Food')
food.show()

#OUTPUT
print('*** Outputs ***')
tip_cheap = fs.FuzzySet('cheap', fs.Trimf([0,5,10]))
tip_avg = fs.FuzzySet('average', fs.Trimf([10,15,20]))
tip_generous = fs.FuzzySet('generous', fs.Trimf([20,25,30]))
tip = fs.FuzzyVariable('tip', [tip_cheap, tip_avg, tip_generous], universe=[0, 30])
print('Tip')
tip.show()

#RULES


ant1  = fs.Antecedent(service['poor'], conector='min')
ant1.add(food['rancid'])
cont1 = fs.Consequent([tip['cheap']])
rule1 = fs.FuzzyRule(ant1, cont1)

ant2  = fs.Antecedent(service['good'] | food['delicious'])
cont2 = fs.Consequent([tip['average']])
rule2 = fs.FuzzyRule(ant2, cont2)

rule3 = fs.FuzzyRule(fs.Antecedent(service['excellent'] | food['delicious']),
                     fs.Consequent([ tip['generous'] ]))

#Building the FIS

fis = fs.FuzzyInferenceSystem([rule1, rule2, rule3],
                              and_op='min',
                              or_op='max')

#FIS Evaluation
inputs = {'service':[8.183, 8.2], 'food':[8.59,4]}
result = fis.eval(inputs, verbose=True)

print("Fuzzy output")
#instances
for i in range(len(inputs['service'])):
    print("Instance: {}\n".format(i))
    result.show(nout=i)