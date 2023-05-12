# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:48:41 2023

@author: 86152
"""
#SYMBOLIC REGRESSION WITH THE EXTRACTED KEY FEATURES(X1,Y1)
from bgp.base import SymbolSet
from sympy.physics.units import kg, m, pa, J, mol, K,W,eV
from bgp.functions.dimfunc import Dim, dless
from bgp.skflow import SymbolLearning
from collections.abc import Iterable

#SR-GP GENETIC PROGRAME
from gplearn.genetic import SymbolicRegressor
for i in range(1,201):
    SR_GP = SymbolicRegressor(population_size=1000,
                              generations=i,
                              tournament_size=20,
                              init_depth=(2, 12),
                              init_method='half and half',
                              function_set=('add', 'sub', 'mul', 'div',
                                            'sqrt','log','abs','neg',
                                            'inv','sin','cos','tan'),
                              random_state=0)
    SR_GP.fit(x1, y1)
    #print ("gene=",i, SR_GP._program)
    SR_GP_predict_y=SR_GP.predict(x1)
    print("gene=",i,r2_score(y1, gSR_GP_predict_y))#0.949230879
    #print(np.sqrt(mean_squared_error(y1, SR_GP_predict_y)))

SR_GP = SymbolicRegressor(population_size=1000,
                              generations=197,
                              tournament_size=20,
                              init_depth=(2, 12),
                              init_method='half and half',
                              function_set=('add', 'sub', 'mul', 'div',
                                            'sqrt','log','abs','neg',
                                            'inv','sin','cos','tan'),
                              random_state=0)
SR_GP.fit(x1, y1)
print (SR_GP._program)
SR_GP_predict_y=SR_GP.predict(x1)
print(r2_score(y1, SR_GP_predict_y))##0.94923087
print(np.sqrt(mean_squared_error(y1, SR_GP_predict_y)))#

#SR-BGP  SymbolLearning #DIMENSIONAL CALCULATION=FALSE
from bgp.base import SymbolSet
from bgp.skflow import SymbolLearning

pset2 = SymbolSet()
pset2.add_features(x1, y1)
pset2.add_operations(power_categories=(2, 1/2, 3, 1/3, 4, 1/4),
                     categories=("Add", "Mul", "Sub", "Div", "exp", "ln", "log", "neg","inv","abs", "sqrt", "sin", "cos","tan"))

SR_BGP= SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=200, re_hall=2, tq=False，
                     add_coef=True, inter_add=True, out_add=True,random_state=1,details=True)
SR_BGP.fit(pset=pset2)
score = SR_BGP.score(x1, y1, "r2")
print(SR_BGP.expr)#0.970385

# SR-DC
Vm_cm3= Dim.convert_to_Dim(1e-6*m**3, unit_system="SI")
Tm_dim=Dim.convert_to_Dim(K, unit_system="SI")
Cp_J_d_K_d_mol= Dim.convert_to_Dim(J/K/mol, unit_system="SI")
Xp_dim=Dim.convert_to_Dim(eV**0.5, unit_system="SI")
Hf_kJ_d_mol=Dim.convert_to_Dim(1000*J/mol, unit_system="SI")
VED_A3=Dim.convert_to_Dim(1e30/m**3, unit_system="SI")
I2_eV=Dim.convert_to_Dim(eV, unit_system="SI")
K_W_d_m_d_K=Dim.convert_to_Dim(W/m/K, unit_system="SI")
E_coh_kJ_d_mol= Dim.convert_to_Dim(1000*J/mol, unit_system="SI")
R_pm= Dim.convert_to_Dim(1e-12*m, unit_system="SI")
Smix_J_d_K_d_mol= Dim.convert_to_Dim(J/K/mol, unit_system="SI")
Eea_eV=Dim.convert_to_Dim(eV, unit_system="SI")

print(Vm_cm3)
print(Tm_dim)
print(Cp_J_d_K_d_mol)
print(Xp_dim)
print(Hf_kJ_d_mol)
print(VED_A3)
print(I2_eV)
print(K_W_d_m_d_K)
print(E_coh_kJ_d_mol)
print(R_pm)
print(Smix_J_d_K_d_mol)
print(Eea_eV)

for i in range(1,7):
    y1_dim = Dim.convert_to_Dim(1e9*pa, unit_system="SI")
    x1_dim = [Vm_cm3[1],Tm_dim[1],Cp_J_d_K_d_mol[1],dless,Xp_dim[1],Hf_kJ_d_mol[1],
          VED_A3[1],dless,I2_eV[1],K_W_d_m_d_K[1]]

    pset1 = SymbolSet()
    pset1.add_features(x1, y1, x_dim=x1_dim, y_dim=y1_dim)
    #pset0.add_constants(c, c_dim=c_dim, c_prob=None)
    pset1.add_operations(power_categories=(2, 1/2, 3, 1/3, 4, 1/4),
                     categories=("Add", "Mul", "Sub", "Div", "exp", "ln", "log", 
                                 "neg","inv","abs", "sqrt", "sin", "cos","tan"))

    SR_DC= SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=200,cal_dim=True, 
                     hall=1, batch_size=40, re_hall=5, mutate_prob=0.8,re_Tree=0,
                     store=False, verbose=True,
                     mate_prob=0.9, max_value=i, initial_min=i, initial_max=i,
                     add_coef=True, inter_add=True,dim_type='integer',
                     limit_type='h_bgp',
                     tq=False,stats={"fitness_dim_max": ["max"], 
                                     "dim_is_target": ["sum"], 
                                     "h_bgp": ["mean"]},
                     personal_map=False, 
                     random_state=1)

    SR_DC.fit(pset=pset1)
    score = SR_DC.score(x1, y1, "r2")
    print(SR_DC.expr)

y1_dim = Dim.convert_to_Dim(1e9*pa, unit_system="SI")
x1_dim = [Vm_cm3[1],Tm_dim[1],Cp_J_d_K_d_mol[1],Hf_kJ_d_mol[1],Xp_dim[1],dless,
          dless,VED_A3[1],I2_eV[1],R_pm[1]]

pset1 = SymbolSet()
pset1.add_features(x1, y1, x_dim=x1_dim, y_dim=y1_dim)
#pset0.add_constants(c, c_dim=c_dim, c_prob=None)
pset1.add_operations(power_categories=(2, 1/2, 3, 1/3, 4, 1/4),
                     categories=("Add", "Mul", "Sub", "Div", "exp", "ln", "log", 
                                 "neg","inv","abs", "sqrt", "sin", "cos","tan"))

SR_DC= SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=200,cal_dim=True, 
                     hall=1, batch_size=40, re_hall=5, mutate_prob=0.8,re_Tree=0, store=False, verbose=True,
                     mate_prob=0.9, max_value=3, initial_min=3, initial_max=3,
                     add_coef=True, inter_add=True,dim_type='integer',limit_type='h_bgp',
                     tq=False,stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
                     personal_map=False, random_state=1)
