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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold


PARSER = argparse.ArgumentParser(description="Software to train ML models for the prediction of selectivity of DAAA reactions using side of addition as target variable.")

PARSER.add_argument("-a", "--algorithm", default='rf', type=str, dest="a", 
                    help="Name of the ML algorithm to be used. Allowed values are: rf for Random Forest, lr for logistic regression, gb for Gradient Boosting")
PARSER.add_argument("-ed", "--electronic_descriptor", default='v2', type=str, dest="ed", 
                    help="Electronic descriptor to be used.  Allowed values are: v1 for Hammet, v2 for Hammet sum, v3 for Hammet avg, v4 for 13C-MNR")

ARGS = PARSER.parse_args()

seed = 2023
print('\nGlobal seed: ', seed)

np.random.seed(seed)


def choose_model(best_params):

    if best_params == None:
        if ARGS.a == 'rf':
            return RandomForestClassifier()
        if ARGS.a == 'lr':
            return LogisticRegression()
        if ARGS.a == 'gb':
            return GradientBoostingClassifier()

    else:
        if ARGS.a == 'rf':
            return RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_leaf=best_params['min_samples_leaf'], 
                                            min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
        if ARGS.a == 'lr':
            return LogisticRegression(penalty=best_params['penalty'], tol=best_params['tol'], C=best_params['C'], 
                                        fit_intercept=best_params['fit_intercept'], random_state=best_params['random_state'],max_iter=best_params['max_iter'])
        if ARGS.a == 'gb':
            return GradientBoostingClassifier(loss = best_params['loss'], learning_rate=best_params['learning_rate'],n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                                min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
  

def hyperparam_tune(X, y, model):

    print('ML algorithm to be tunned:', str(model))

    if str(model) == 'LogisticRegression()':
        hyperP = dict(penalty=['l2'], 
                        tol=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                        C=[.001, .005, .01, .05, .1, .5, 1, ],
                        fit_intercept=[False, True],
                        random_state = [seed], 
                        max_iter=[1000])
    
    
    elif str(model) == 'RandomForestClassifier()':
        hyperP = dict(n_estimators=[100, 300, 500, 800], 
                        max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2, 5, 10, 15, 100],
                        min_samples_leaf=[1, 2, 5, 10],
                        random_state = [seed])

    elif str(model) == 'GradientBoostingClassifier()':
        hyperP = dict(loss=['log_loss'], learning_rate=[0.1, 0.2, 0.3],
                        n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2],
                        min_samples_leaf=[1, 2],
                        random_state = [seed])

    gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
    bestP = gridF.fit(X, y)

    params = bestP.best_params_
    print('Best hyperparameters:', params, '\n')

    return params



def electronic_descriptor():
    if ARGS.ed == 'v1':
        return 'A(Hammett)'
    if ARGS.ed == 'v2':
        return 'A(Hammett sum)'
    if ARGS.ed == 'v3':
        return 'A(Hammett avg.)'
    if ARGS.ed == 'v4':
        return 'A(C-NMR-ipso-shifts)'


def choose_dataset():

    descriptors = [
    electronic_descriptor(),
    'A(stout)', 'B(volume)', 'B(Hammett)',  'C(volume)', 'C(Hammett)', 'D(volume)', 'D(Hammett)', 'UL(volume)', 'LL(volume)', 'UR(volume)', 'LR(volume)', 'dielectric constant', '%topA']

    data = pd.read_csv('DAAA-Final.csv')

    data = data.filter(descriptors)

    #remove erroneous data
    data = data.dropna(axis=0)


    X = data.drop(['%topA'], axis = 1)
    X = RobustScaler().fit_transform(np.array(X))

    y = np.where(data['%topA']<50, 0, 1)

    print('Features shape: ', X.shape)
    print('Y target variable shape: ' , y.shape)

    return X, y, descriptors


def main():

    random_seeds = np.random.randint(0, high=1001, size=30) 
    print('Random seeds generated for training-test random spliting:')
    print(random_seeds)
    print('\n')

    X, y, feat_names = choose_dataset()

    print('The selected descriptors are: ', feat_names)

    best_params = hyperparam_tune(X, y, choose_model(best_params=None)) #hyperparameter tuning completed on whole subset


    auroc_cv_scores = []
    prec_cv_scores = []
    recall_cv_scores = []
    auroc_val_scores = []
    prec_val_scores = []
    recall_val_scores = []

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

                auroc = roc_auc_score(y_train[test], predictions)
                prec = precision_score(y_train[test], predictions)
                recall = recall_score(y_train[test], predictions)
                auroc_cv_scores.append(auroc)
                prec_cv_scores.append(prec)
                recall_cv_scores.append(recall)


        # predict on validaiton set
        model = choose_model(best_params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        auroc = roc_auc_score(y_val, predictions)
        prec = precision_score(y_val, predictions)
        recall = recall_score(y_val, predictions)
        auroc_val_scores.append(auroc)
        prec_val_scores.append(prec)
        recall_val_scores.append(recall)


    print('Model:',  model)
    print('Random Seeds: ', random_seeds, '\n')

    print('Num CV Scores: ', len(auroc_cv_scores))
    print('CV AUROC Mean: ', round(np.mean(np.array(auroc_cv_scores)),3), '+/-', round(np.std(np.array(auroc_cv_scores)),3))
    print('CV precision Mean %: ', round(np.mean(np.array(prec_cv_scores)),3), '+/-', round(np.std(np.array(prec_cv_scores)),3))
    print('CV recall Mean: ', round(np.mean(np.array(recall_cv_scores)),3), '+/-', round(np.std(np.array(recall_cv_scores)),3), '\n')


    print('Num Val Scores: ', len(auroc_val_scores))
    print('Val AUROC Mean: ', round(np.mean(np.array(auroc_val_scores)),3), '+/-', round(np.std(np.array(auroc_val_scores)),3))
    print('Val precision Mean %: ', round(np.mean(np.array(prec_val_scores)),3), '+/-',round(np.std(np.array(prec_val_scores)),3))
    print('Val recall Mean: ', round(np.mean(np.array(recall_val_scores)),3), '+/-', round(np.std(np.array(recall_val_scores)),3), '\n')


if __name__ == "__main__":  
    main()   