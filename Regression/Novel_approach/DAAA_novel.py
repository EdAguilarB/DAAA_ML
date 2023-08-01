import argparse
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
from matplotlib import cm
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

import shap


PARSER = argparse.ArgumentParser(description="Software to train ML models for the prediction of selectivity of DAAA reactions.")

PARSER.add_argument("-a", "--algorithm", default='rf', type=str, dest="a", 
                    help="Name of the ML algorithm to be used. Allowed values are: rf for Random Forest, lr for linear regression, gb for Gradient Boosting")
PARSER.add_argument("-ed", "--electronic_descriptor", default='v2', type=str, dest="ed", 
                    help="Electronic descriptor to be used.  Allowed values are: v1 for Hammet, v2 for Hammet sum, v3 for Hammet avg, v4 for 13C-MNR")

ARGS = PARSER.parse_args()

seed = 2023
print('\nGlobal seed: ', seed)

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
  

def hyperparam_tune(X, y, model):

    print('ML algorithm to be tunned:', str(model))


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
    y = data['%topA']
    print('Features shape: ', X.shape)
    print('Y target variable shape: ' , y.shape)

    return X, y, descriptors


def plot_regression(y, y_hat, figure_title, dependent_variable):
    fig, ax = plt.subplots()
    ax.scatter(y, y_hat)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured ' + dependent_variable, fontsize = 13)
    ax.set_ylabel('Predicted ' + dependent_variable, fontsize = 13)
    plt.title(figure_title, fontsize = 13)
    coefficient_of_dermination = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y,y_hat)
    legend = '$R^2$: '+str(float("{0:.2f}".format(coefficient_of_dermination))) + '\nRMSE: ' + str(float("{0:.2f}".format(rmse))) + '\nMAE: ' + str(float("{0:.2f}".format(mae)))
    plt.legend(['Best fit',legend],loc = 'upper left', fontsize = 13)
    plt.show()



def main():

    random_seeds = np.random.randint(0, high=1001, size=30) 
    print('Random seeds generated for training-test random spliting:')
    print(random_seeds)
    print('\n')

    X, y, feat_names = choose_dataset()

    print('The selected descriptors are: ', feat_names)

    best_params = hyperparam_tune(X, y, choose_model(best_params=None)) #hyperparameter tuning completed on whole subset


    r2_cv_scores = []
    rmse_cv_scores = []
    mae_cv_scores = []
    r2_val_scores = []
    rmse_val_scores = []
    mae_val_scores = []

    shap_analysis = np.random.randint(0, len(random_seeds)-1)
    print(f'Random partition seed used to make SHAP analysis: {random_seeds[shap_analysis]}')

    print('Initialising training-test spliting and training-testing process.')
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

        if i == shap_analysis:

            model_shap = copy.deepcopy(model)

            X_shap_train = copy.deepcopy(X_train)
            y_shap_train = copy.deepcopy(y_train)

            X_shap_test = copy.deepcopy(X_val)
            y_shap_test = copy.deepcopy(y_val)


    print('Model:',  model)
    print('Random Seeds: ', random_seeds, '\n')

    print('Num CV Scores: ', len(r2_cv_scores))
    print('CV R2 Mean: ', round(np.mean(np.array(r2_cv_scores)),3), '+/-', round(np.std(np.array(r2_cv_scores)),3))
    print('CV RMSE Mean %: ', round(np.mean(np.array(rmse_cv_scores)),2), '+/-', round(np.std(np.array(rmse_cv_scores)),2))
    print('CV MAE Mean: ', round(np.mean(np.array(mae_cv_scores)),2), '+/-', round(np.std(np.array(mae_cv_scores)),2), '\n')


    print('Num Val Scores: ', len(r2_val_scores))
    print('Val R2 Mean: ', round(np.mean(np.array(r2_val_scores)),3), '+/-', round(np.std(np.array(r2_val_scores)),3))
    print('Val RMSE Mean %: ', round(np.mean(np.array(rmse_val_scores)),2), '+/-',round(np.std(np.array(rmse_val_scores)),3))
    print('Val MAE Mean: ', round(np.mean(np.array(mae_val_scores)),2), '+/-', round(np.std(np.array(mae_val_scores)),2), '\n')

    print('Initialising SHAP analysis')

    y_hat = model_shap.predict(X_shap_train)
    plot_regression(y_shap_train, y_hat, "Results for the Training Set", '%top')

    y_hat = model_shap.predict(X_shap_test)
    plot_regression(y_shap_test, y_hat, "Results for the Test Set", '%top')

    if ARGS.a == 'lr':
        explainer = shap.LinearExplainer(model, X_shap_train)
    else:
        explainer = shap.Explainer(model_shap, output_names=feat_names)

    shap_values_test = explainer.shap_values(X_shap_test)
    shap_values_train = explainer.shap_values(X_shap_train)

    print(f'Ploting shap analysis for {str(model_shap)}:\n')
    plt.figure()
    shap.summary_plot(shap_values_train, X_train, plot_type="bar", max_display = 8, plot_size= (14,5.5), feature_names=feat_names) 

    plt.figure()
    shap.summary_plot(shap_values_train, X_train, max_display = 8, color_bar_label = 'Descriptor value', show = False, plot_size= (14,5.5), feature_names=feat_names)
    plt.grid()
    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()

    plt.gcf().axes[-1].set_box_aspect(50)
    #Changing plot colours
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(cm.get_cmap('coolwarm'))
    plt.show()


if __name__ == "__main__":  
    main()   