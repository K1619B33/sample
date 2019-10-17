import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')
quality.automf(3)
service.automf(3)

tip['general'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['convoy'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['Ambulance'] = fuzz.trimf(tip.universe, [13, 25, 25])

quality['average'].view()
service.view()
tip.view()
rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['general'])
rule2 = ctrl.Rule(service['average'], tip['convoy'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['Ambulance'])

rule1.view()
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8


tipping.compute()
print(tipping.output['tip'])
tip.view(sim=tipping)
