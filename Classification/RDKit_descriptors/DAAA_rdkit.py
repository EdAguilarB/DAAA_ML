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
from sklearn.feature_selection import RFECV

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Fragments


PARSER = argparse.ArgumentParser(description="Software to train ML models for the prediction of selectivity of DAAA reactions using traditional descriptors.")

PARSER.add_argument("-a", "--algorithm", default='rf', type=str, dest="a", 
                    help="Name of the ML algorithm to be used. Allowed values are: rf for Random Forest, lr for linear regression, gb for Gradient Boosting")

ARGS = PARSER.parse_args()

#set a global seed for the software
seed = 2023
print('\n Global seed: ', seed)
np.random.seed(seed)

#chooses a model given a dictionary of hyperparameters 
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
        

#opens the dataset 
def choose_dataset():
    cols = ['Example', 'SMILES', 'ligand_smiles', 'solvent SMILE', '%topA']
    data = pd.read_csv('DAAA-Final.csv', index_col=0, usecols=cols)
    data = data.dropna(axis=0)
    return data

#counts fragments given a molecule and a molecule type
def count_fragments(mol, m_type):
    mol_frags = {}
    for i in dir(Fragments):
        if 'fr' in i:
            item = getattr(Fragments,i)
            if callable(item):
                mol_frags[i+'_'+m_type] = item(mol)
    return mol_frags


#calculates molecular descriptors given a molecule and a molecule type
def calc_descriptors(mol, m_type):
    descriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex','qed','MaxPartialCharge','MinPartialCharge',
               'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
               'HallKierAlpha', 'TPSA', 'MolLogP', 'MolMR']
    mol_desc = {}
    for i in dir(Descriptors):
        if i in descriptors:
            item = getattr(Descriptors,i)
            if callable(item):
                mol_desc[i+'_'+m_type] = item(mol)
    return mol_desc


#calculates fragments and molecular descriptors given the dataset
def descriptors_all():

    data = choose_dataset()

    #count fragments of substrate
    frags_all = pd.DataFrame()
    for index, row in data.iterrows():
        smiles = row['SMILES']
        frags = pd.DataFrame(count_fragments(Chem.MolFromSmiles(smiles), 'substrate'), index=[index])
        frags_all = pd.concat([frags_all, frags], axis= 0)
    frags_all = frags_all.loc[:, (frags_all != 0).any(axis=0)]

    #calculate descriptors of ligands
    desc_all = pd.DataFrame()
    for index, row in data.iterrows():
        smiles = row['ligand_smiles']
        desc = pd.DataFrame(calc_descriptors(Chem.MolFromSmiles(smiles), 'ligand'), index=[index])
        desc_all = pd.concat([desc_all, desc], axis= 0)
    desc_all = desc_all.loc[:, (desc_all != 0).any(axis=0)]

    #calculate Hall-Kier Alpha of solvent
    X = pd.concat([frags_all, desc_all], axis=1)
    X['HA_solv'] = data['solvent SMILE'].apply(lambda m: Descriptors.HallKierAlpha(Chem.MolFromSmiles(m)))

    #get features names
    feat_names = X.columns

    #scale features and convert to numpy
    X = RobustScaler().fit_transform(np.array(X))

    y = np.where(data['%topA']<50, 0, 1)

    print('Features shape: ', X.shape)
    print('Target variable shape: ', y.shape)

    return X, y, feat_names


#selects the best features for predicting given a ML estimator
def select_features(model, X, y, names):
    print('Using ', model, 'as estimator of importance.')
    rfecv = RFECV(estimator = model,
                step=1,
                cv=10,
                scoring='neg_root_mean_squared_error',
                min_features_to_select=1)
    rfecv.fit(X, y)
    print(f"Optimal number of features: {rfecv.n_features_}")
    names = rfecv.get_feature_names_out(names)
    print('The selected features are:')
    print(names, '\n')
    X = rfecv.transform(X)
    return X


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


def main():

    random_seeds = np.random.randint(0, high=1001, size=30) 
    print('Random seeds generated for training-test random spliting:')
    print(random_seeds)
    print('\n')

    X, y, feat_names = descriptors_all()

    best_params = hyperparam_tune(X, y, choose_model(best_params=None))

    X = select_features(choose_model(best_params=best_params),X,y,names=feat_names)

    auroc_cv_scores = []
    prec_cv_scores = []
    recall_cv_scores = []
    auroc_val_scores = []
    prec_val_scores = []
    recall_val_scores = []

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
        for _ in range(3):
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