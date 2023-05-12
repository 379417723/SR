# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:06:15 2023

@author: 86152
"""
#KEY FEATURES EXTRACTION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
import scipy as sp
%matplotlib inline
%config inlinebackend.figure_format="retina"
from matplotlib.font_manager import FontProperties
fonts=FontProperties(fname="/library/fonts/华文细黑.ttf",size=14)
from mpl_toolkits.mplot3d import axes3d
from sklearn import tree,metrics
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.model_selection import LeaveOneOut
from collections.abc import Iterable

Strength=pd.read_csv("J:/xxxx/Dataset.csv")
X=['Smix','VEC','Tm','△Tm','χp','△χp','χc','△χc','χm','△χm','Vm','R',
   'δ','E_coh','Cp','I1','I2','Eea','AW','K','Hf','VED']
Y="σy(GPa)"
x=np.array(Strength[X])
y=np.array(Strength[Y])
print(x.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()
RF=RandomForestRegressor(max_depth= max_depth, max_features=max_features,
			min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
			n_estimators= n_estimators)#HYPERPARAMETERS BAYES_OPT
RF_predict=cross_val_predict(RF, x, y, cv=loocv)
print(r2_score(y, RF_predict))
importanceRF=RF.fit(x,y).feature_importances_
print(importanceRF)
#import numpy
#numpy.savetxt('importanceRF.csv',importanceRF, delimiter = ',',header=",") 

import shap
RFmodel=RF.fit(x,y)
explainer = shap.TreeExplainer(RFmodel)
shapRF= explainer(Strength[X])

fig = plt.figure()
shap.summary_plot(shapRF, Strength[X], show = False)
#plt.savefig('shap.pdf')

shap_values = explainer.shap_values(Strength[X])
import numpy
numpy.savetxt('shap_values.csv',shap_values, delimiter = ',')
numpy.savetxt('x.csv',Strength[X], delimiter = ',')