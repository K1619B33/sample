import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
imp = ctrl.Consequent(np.arange(0, 26, 1), 'imp')
# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)
# Custom membership functions can be built interactively with a familiar,
# Pythonic API
imp['general'] = fuzz.trimf(imp.universe, [0, 0, 13])
imp['convoy'] = fuzz.trimf(imp.universe, [0, 13, 25])
imp['ambulance'] = fuzz.trimf(imp.universe, [13, 25, 25])
quality['wait'].view()
service.view()
tip.view()
rule1 = ctrl.Rule(quality['stop'] | service['stop'], imp['general'])
rule2 = ctrl.Rule(service['wait'], imp['convoy'])
rule3 = ctrl.Rule(service['go'] | quality['go'], imp['ambulance'])
rule1.view()
imprtnt_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
imprtnt = ctrl.ControlSystemSimulation(imprtnt_ctrl)
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
imprtnt.input['quality'] = 6.5
imprtnt.input['service'] = 9.8

# Crunch the numbers
imprtnt.compute()
print(tipping.output['imp'])
imp.view(sim=imprtnt)
