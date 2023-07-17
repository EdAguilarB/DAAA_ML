# Copyright 2021 Benjamin Owen, Apache 2.0
# University of Nottingham, GSK-EPSRC Prosperity Partnership EP/S035990/1
# and EPSRC award EP/S022236/1 

import argparse
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold


PARSER = argparse.ArgumentParser(description="Software to train ML models for the prediction of selectivity of DAAA reactions.")

PARSER.add_argument("-a", "--algorithm", default='rf', type=str, dest="a", 
                    help="Name of the ML algorithm to be used. Allowed values are: rf for Random Forest, lr for linear regression, gb for Gradient Boosting")
PARSER.add_argument("-ed", "--electronic_descriptor", default='v2', type=str, dest="ed", 
                    help="Electronic descriptor to be used.  Allowed values are: v1 for Hammet, v2 for Hammet sum, v3 for Hammet avg, v4 for 13C-MNR")

ARGS = PARSER.parse_args()

seed = 2023
print('Global seed: ', seed)

np.random.seed(seed)


def choose_model(best_params):

    if best_params == None:
        if ARGS.a == 'rf':
            return RandomForestRegressor()
        if ARGS.a == 'lr':
            return LinearRegression()
        if ARGS.a == 'gb':
            return GradientBoostingRegressor()

    else:
        if ARGS.a == 'rf':
            return RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_leaf=best_params['min_samples_leaf'], 
                                     min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
        if ARGS.a == 'lr':
            return LinearRegression()
        if ARGS.a == 'gb':
            return GradientBoostingRegressor(loss = best_params['loss'], learning_rate=best_params['learning_rate'],n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                             min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
 


def choose_dataset():
    return 'DAAA-Final'

  

def hyperparam_tune(X, y, model):

    print(str(model))

    if str(model) == 'LinearRegression()':
        return None
    
    else: 
        if str(model) == 'RandomForestRegressor()':
            hyperP = dict(n_estimators=[100, 300, 500, 800], 
                        max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2, 5, 10, 15, 100],
                        min_samples_leaf=[1, 2, 5, 10],
                        random_state = [seed])

        elif str(model) == 'GradientBoostingRegressor()':
            hyperP = dict(loss=['squared_error'], learning_rate=[0.1, 0.2, 0.3],
                        n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2],
                        min_samples_leaf=[1, 2],
                        random_state = [seed])

        gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
        bestP = gridF.fit(X, y)
        print(bestP.best_params_)
        return bestP.best_params_

random_seeds = np.random.randint(0, high=1001, size=30)

def electronic_descriptor():
    if ARGS.ed == 'v1':
        return 'A(Hammett)'
    if ARGS.ed == 'v2':
        return 'A(Hammett sum)'
    if ARGS.ed == 'v3':
        return 'A(Hammett avg.)'
    if ARGS.ed == 'v4':
        return 'A(C-NMR-ipso-shifts)'

descriptors = [

electronic_descriptor(),

'A(stout)', 'B(volume)', 'B(Hammett)',  'C(volume)', 'C(Hammett)', 'D(volume)', 'D(Hammett)', 'UL(volume)', 'LL(volume)', 'UR(volume)', 'LR(volume)', 'dielectric constant', '%topA']

data = pd.read_csv('DAAA-Final.csv')

data = data.filter(descriptors)

#remove erroneous data
data = data.dropna(axis=0)


X = data.drop(['%topA'], axis = 1)
X = RobustScaler().fit_transform(np.array(X))
y = data['%topA']
print('Features shape: ', X.shape)
print('Y target variable shape: ' , y.shape)

best_params = hyperparam_tune(X, y, choose_model(best_params=None)) #hyperparameter tuning completed on whole subset
#print(best_params)

#best_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#best_params = {'learning_rate': 0.3, 'loss': 'squared_error', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
#best_params = None#for LR

r2_cv_scores = []
rmse_cv_scores = []
mae_cv_scores = []
r2_val_scores = []
rmse_val_scores = []
mae_val_scores = []

for i in range(len(random_seeds)):
    # split into training and validation sets, 9:1
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.1, random_state=random_seeds[i])

    X_train = np.array(X_train).astype('float64')
    X_val = np.array(X_val).astype('float64')
    y_train = np.array(y_train).astype('float64')
    y_val = np.array(y_val).astype('float64')

    # 5 fold CV on training set, repeated 3 times
    for j in range(3):
        kfold = KFold(n_splits=5)
        for train, test in kfold.split(X_train, y_train):
            model = choose_model(best_params)
            model.fit(X_train[train], y_train[train])
            predictions = model.predict(X_train[test]).reshape(1, -1)[0]

            r2 = r2_score(y_train[test], predictions)
            rmse = math.sqrt(mean_squared_error(y_train[test], predictions))
            mae = mean_absolute_error(y_train[test], predictions)
            r2_cv_scores.append(r2)
            rmse_cv_scores.append(rmse)
            mae_cv_scores.append(mae)


    # predict on validaiton set
    model = choose_model(best_params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    r2 = r2_score(y_val, predictions)
    rmse = math.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2_val_scores.append(r2)
    rmse_val_scores.append(rmse)
    mae_val_scores.append(mae)


print('Model:',  model)
print('Data Subset: ',  choose_dataset())
print('Random Seeds: ', random_seeds, '\n')

print('Num CV Scores: ', len(r2_cv_scores))
print('CV R2 Mean: ', round(np.mean(np.array(r2_cv_scores)),3), '+/-', round(np.std(np.array(r2_cv_scores)),3))
print('CV RMSE Mean %: ', round(np.mean(np.array(rmse_cv_scores)),2), '+/-', round(np.std(np.array(rmse_cv_scores)),2))
print('CV MAE Mean: ', round(np.mean(np.array(mae_cv_scores)),2), '+/-', round(np.std(np.array(mae_cv_scores)),2), '\n')


print('Num Val Scores: ', len(r2_val_scores))
#print(r2_val_scores)
print('Val R2 Mean: ', round(np.mean(np.array(r2_val_scores)),3), '+/-', round(np.std(np.array(r2_val_scores)),3))
print('Val RMSE Mean %: ', round(np.mean(np.array(rmse_val_scores)),2), '+/-',round(np.std(np.array(rmse_val_scores)),3))
print('Val MAE Mean: ', round(np.mean(np.array(mae_val_scores)),2), '+/-', round(np.std(np.array(mae_val_scores)),2))

data = [r2_val_scores]
df = pd.DataFrame(data)
#print (df.T)
#df.T.to_excel(r'C:\Users\Declan\Desktop\V1-RFR.xlsx')