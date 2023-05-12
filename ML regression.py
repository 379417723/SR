# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:11:06 2023

@author: 86152
"""
#ML REGRESSION ALGORITHMS ANN,RF,XGB,SVR,LASSO,RIDGE
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

ANN=MLPRegressor()
hidden_layer_sizes=[(100,), (100, 100), (150, 150)]
solver= ['adam', 'sgd', 'lbfgs']
alpha=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
para_grid=[{"hidden_layer_sizes":hidden_layer_sizes,"solver":solver,"alpha":alpha}]
ANN_GridSearch=GridSearchCV(estimator=ANN21,param_grid=para_grid,cv=5,n_jobs=4)
ANN_GridSearch.fit(x1_std, y1)

print("Best_params：",ANN_GridSearch.best_params_)
print("Best_score：",ANN_GridSearch.best_score_)
print("Best_estimator：",ANN_GridSearch.best_estimator_)

from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
def RF_cv(n_estimators, max_depth, min_samples_split,min_samples_leaf,max_features):
    RFreg=RandomForestRegressor(n_estimators=int(n_estimators),
                             max_depth=int(max_depth),
                             min_samples_split=int(min_samples_split),
                             min_samples_leaf=int(min_samples_leaf),
                             max_features=int(max_features),
                             random_state=2)                      
    RF_predict=cross_val_predict(RFreg, x1, y1, cv=10) 
    res = r2_score(y1, RF_predict)
    return res

RF_op = BayesianOptimization(
        RF_cv,
        {'n_estimators': (100,1000), 
         'max_depth':(3,100),
         'min_samples_split':(2,100),
         'min_samples_leaf':(1, 100),
         'max_features':(1, 10)}
        )

RF_op.maximize()

print(RF_op.max)


import xgboost as xgb
XGB=xgb.XGBRegressor()#default hyperprameters
XGB_predict=cross_val_predict(XGB, x1, y1, cv=loocv)
print(r2_score(y1, XGB_predict))

from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor(max_depth= 100, max_features=10,min_samples_leaf=1, min_samples_split=2, n_estimators= 982)
RFR_predict=cross_val_predict(RFR, x1, y1, cv=loocv)
print(r2_score(y1, RFR_predict))

from sklearn.svm import SVR
x1_std=StandardScaler().fit_transform(x1)
SVR_model=SVR(kernel='rbf',C=10, gamma=0.1)
SVR_predict=cross_val_predict(SVR_model, x1_std, y1, cv=loocv)
print(r2_score(y1,SVR_predict))
print(np.sqrt(mean_squared_error(y1,SVR_predict)))

from sklearn.neural_network import MLPRegressor
ANN=MLPRegressor(alpha=0.5, hidden_layer_sizes=(100, 100), solver= 'lbfgs')
ANN_predict=cross_val_predict(ANN, x1_std, y1, cv=loocv)
print(r2_score(y1, ANN_predict))
print(np.sqrt(mean_squared_error(y1, ANN_predict)))

def lasso_regression(x,y,alpha):
    lasso_reg=Lasso(alpha=alpha,normalize=True,fit_intercept=True)
    y_pred_lasso=cross_val_predict(lasso_reg, x, y, cv=loocv)
    lasso_model=lasso_reg.fit(x,y) 

    ret=[alpha]
    ret.append(r2_score(y,y_pred_lasso))#r2_score
    ret.append(mean_squared_error(y,y_pred_lasso))
    ret.append(sqrt(mean_squared_error(y,y_pred_lasso)))
    ret.extend([lasso_model.intercept_])
    ret.extend(lasso_model.coef_)#
    return ret

alpha_lasso=(0.00000001,1.25893E-08,1.58489E-08,1.99526E-08,2.51189E-08,3.16228E-08,3.98107E-08,5.01187E-08,6.30957E-08,7.94328E-08,0.0000001,1.25893E-07,1.58489E-07,1.99526E-07,2.51189E-07,3.16228E-07,3.98107E-07,5.01187E-07,6.30957E-07,7.94328E-07,1E-06,1.25893E-06,1.58489E-06,1.99526E-06,2.51189E-06,3.16228E-06,3.98107E-06,5.01187E-06,6.30957E-06,7.94328E-06,0.00001,1.25893E-05,1.58489E-05,1.99526E-05,2.51189E-05,3.16228E-05,3.98107E-05,5.01187E-05,6.30957E-05,7.94328E-05,0.0001,0.000125893,0.000158489,0.000199526,0.000251189,0.000316228,0.000398107,0.000501187,0.000630957,0.000794328,0.001,0.001258925,0.001584893,0.001995262,0.002511886,0.003162278,0.003981072,0.005011872,0.006309573,0.007943282,0.01,0.012589254,0.015848932,0.019952623,0.025118864,0.031622777,0.039810717,0.050118723,0.063095734,0.079432823,0.1,0.125892541,0.158489319,0.199526231,0.251188643,0.316227766,0.398107171,0.501187234,0.630957344,0.794328235,1,1.258925412,1.584893192,1.995262315,2.511886432,3.16227766,3.981071706,5.011872336,6.309573445,7.943282347,10,12.58925412,15.84893192,19.95262315,25.11886432,31.6227766,39.81071706,50.11872336,63.09573445,79.43282347,100,316.227766,1000,3162.27766,10000,31622.7766,100000,316227.766,1000000,3162277.66,10000000,31622776.6,100000000)#np.linspace(0,0.02,600)

col=['alpha','R2_score','MSE','RMSE','intercept','Vm','Tm','Cp','△χp','χp','Hf','VED','δ','I2','K']
ind=['alpha_%.2g'%alpha_lasso[i] for i in range(0,len(alpha_lasso))]
lasso_coef=pd.DataFrame(index=ind,columns=col)

for i in range(len(alpha_lasso)):
    lasso_coef.iloc[i,]=lasso_regression(x1,y1,alpha_lasso[i])
import numpy
numpy.savetxt('lasso_coef.csv',lasso_coef, delimiter = ',',header=",".join(col)) 
#coef_G.sort_values("alpha").head(5)

def ridge_regression(x,y,alpha):
    ridge_reg=Ridge(alpha=alpha,normalize=True,fit_intercept=True)#
    y_pred_ridge=cross_val_predict(ridge_reg, x, y, cv=loocv)#
    ridge_model=ridge_reg.fit(x,y) #

    ret=[alpha]
    ret.append(r2_score(y,y_pred_ridge))#
    ret.append(mean_squared_error(y,y_pred_ridge))#
    ret.append(sqrt(mean_squared_error(y,y_pred_ridge)))#
    ret.extend([ridge_model.intercept_])#
    ret.extend(ridge_model.coef_)#
    return ret

alpha_ridge=(0.00000001,1.25893E-08,1.58489E-08,1.99526E-08,2.51189E-08,3.16228E-08,3.98107E-08,5.01187E-08,6.30957E-08,7.94328E-08,0.0000001,1.25893E-07,1.58489E-07,1.99526E-07,2.51189E-07,3.16228E-07,3.98107E-07,5.01187E-07,6.30957E-07,7.94328E-07,1E-06,1.25893E-06,1.58489E-06,1.99526E-06,2.51189E-06,3.16228E-06,3.98107E-06,5.01187E-06,6.30957E-06,7.94328E-06,0.00001,1.25893E-05,1.58489E-05,1.99526E-05,2.51189E-05,3.16228E-05,3.98107E-05,5.01187E-05,6.30957E-05,7.94328E-05,0.0001,0.000125893,0.000158489,0.000199526,0.000251189,0.000316228,0.000398107,0.000501187,0.000630957,0.000794328,0.001,0.001258925,0.001584893,0.001995262,0.002511886,0.003162278,0.003981072,0.005011872,0.006309573,0.007943282,0.01,0.012589254,0.015848932,0.019952623,0.025118864,0.031622777,0.039810717,0.050118723,0.063095734,0.079432823,0.1,0.125892541,0.158489319,0.199526231,0.251188643,0.316227766,0.398107171,0.501187234,0.630957344,0.794328235,1,1.258925412,1.584893192,1.995262315,2.511886432,3.16227766,3.981071706,5.011872336,6.309573445,7.943282347,10,12.58925412,15.84893192,19.95262315,25.11886432,31.6227766,39.81071706,50.11872336,63.09573445,79.43282347,100,316.227766,1000,3162.27766,10000,31622.7766,100000,316227.766,1000000,3162277.66,10000000,31622776.6,100000000)#np.linspace(0,0.02,600)

col=['alpha','R2_score','MSE','RMSE','intercept','Vm','Tm','Cp','△χp','χp','Hf','VED','δ','I2','K']
ind=['alpha_%.2g'%alpha_ridge[i] for i in range(0,len(alpha_ridge))]
ridge_coef=pd.DataFrame(index=ind,columns=col)

for i in range(len(alpha_ridge)):
    ridge_coef.iloc[i,]=ridge_regression(x1,y1,alpha_ridge[i])
import numpy
numpy.savetxt('ridge_coef.csv',ridge_coef, delimiter = ',',header=",".join(col)) 
#coef_G.sort_values("alpha").head(5)