# rayidsplitter
from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time

def get_mse(predicted, val_targets):
    return (((predicted - val_targets) ** 2).sum()) / len(predicted)

def splitter(outcome_variable, dataset, models_to_run = ['RFR', 'DTR', 'KNNR', 'KNN', 'DT', 'RF','LR','ET','AB','GB'],
                  cols=None, testsize=0.20, 
                  results_dataframe=None, regress_only=False,
                  filename='output'):
    
    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'RFR': RandomForestRegressor(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'DTR': DecisionTreeRegressor(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'KNNR': KNeighborsRegressor(n_neighbors=3),
            }
      
    grid = {
    'RF': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'RFR': {'n_estimators': [1,10,100,1000,10000], 'min_samples_split': [2, 5, 10]},
    'DTR' : {'max_depth': [3, 10, 15, 50, 100], 'min_samples_split':[2, 5, 10, 50, 100]},
    'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'KNNR' : {'n_neighbors': [1,5,10,25,50,100], 'weights': ['uniform', 'distance'], 'algorithm': ['auto','ball_tree','kd_tree']}
           }
    # X_train, X_val = cross_validation.train_test_split(train_transformed, test_size = 0.20)
    results_matrix = {'Model': [], 
                      'Training_set': [], 
                      'Y_outcome' : [],
                      'Test_size': [],
                      'Classifier': [],
                      'Regressor': [],
                      'Predictors': [],
                      'n_estimators': [],
                      'max_depth': [],
                      'min_samples_split': [],
                      'metric': [],
                      'metric_score': [],
                      'cross_val_metric': [],
                      'score': []
                     }
    write_to = filename + '.xlsx'
        
    # Loop through models in clfs
    model_number = 0
    best = {'score': float(0), 'model_dict': None, 'Classifier': None, 'Regressor': None}
    if not results_dataframe:
        results_dataframe = results_matrix.copy()
    # Split the training data into a training set and a validation set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset[cols], dataset[outcome_variable], test_size = testsize)   
    
    print('regress_only?', regress_only)
    model_number = 0
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        model_dict = {}
        print(models_to_run[index])
        learner = models_to_run[index]
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                print(clf)
            except:
                continue
            for k in results_dataframe.keys():
                results_dataframe[k].append(np.nan)
                print(k, results_dataframe[k])
                model_dict[k] =  np.nan
            if not regress_only:
                try:
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    print(precision_at_k(y_test,y_pred_probs,.05))
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
                except:
                    pass
            
            #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
            #print threshold
            # print(precision_at_k(y_test,y_pred_probs,.05))
            # plot_precision_recall_n(y_test,y_pred_probs,clf)
            model_dict['Model'] = model_number
            model_dict['Y_outcome'] = outcome_variable
            model_dict['Training_set'] = dataset
            model_dict['Test_size'] = testsize 
            model_dict['Classifier'] = learner       

            #Fit the model to the training inputs and training targets
            model_dict['Predictors'] = cols
            clf.fit(X_train.as_matrix(), y_train.as_matrix())

            #Predict the output on the validation
            predicted = clf.predict(X_test.as_matrix())
            
            mse = get_mse(predicted, y_test.as_matrix())
             
            model_dict['metric_score'] = mse
                
            score = np.nan
            
            try:
                score = clf.score(X_train, y_train, sample_weight=None)
                model_dict['score'] = score
                score = model_dict['score']
            except:
                print('Clf gave no score: {}.\nSCORE is :{}'.format(clf, score))
            try:
                scores = cross_val_score(clf, X_train, y_train)
                model_dict['cross_val_metric'] = 'cross_val_score'
                model_dict['score'] = scores
                score = scores
            except:
                model_dict['cross_val_metric'] = 'NA'
            model_dict['score'] = score
            for k in results_dataframe:
                print(k, results_dataframe[k])
            for k in model_dict:
                print(k, model_dict[k])
            
            # Sweep up any new or unused results keys into the results dictionary.
            for element in model_dict:
                if element not in results_dataframe:
                    results_dataframe[element] = []
                    for i in range(model_number - len(results_dataframe[element])):
                        results_dataframe[element].append(np.nan)
                    results_dataframe[element].append(model_dict[element])

            for unused_k in results_dataframe:
                if unused_k not in model_dict:
                    for i in range(model_number - len(results_dataframe[unused_k])):
                        results_dataframe[unused_k].append(np.nan)
                else:
                    results_dataframe[unused_k].append(model_dict[unused_k])
            results = pd.DataFrame.from_dict(results_dataframe)
            # results.to_excel(write_to)
            
            if model_dict['score'] > best['score']:
                print('\nMODEL SCORE to beat:', best['score'])
                best['model_dict'] = None
                best['score'] = model_dict['score']
                best['Classifier'] = model_dict['Classifier']
                print('\n\tBETTER MODEL!\n')
                print('Model {}.'.format(model_dict['Model']))
                for better_result in model_dict:
                    if better_result != 'Model':
                        print(better_result, model_dict[better_result])
                best['model_dict'] = model_dict.copy()
                best['PARAMETERS'] = clf.get_params()
                best['model_dict'] = model_dict.copy()
                try:
                    importances = clf.feature_importances_
                    sorted_idx = np.argsort(importances)
                    padding = np.arange(len(cols)) + 0.5
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(10,8))
                    t = 'Model {} Imputed {} by {}:\nVariable Importances of {}'.format(model_number, outcome_variable, learner, cols)
                    doc = '{}_{}_{}_feat_importance.png'.format(filename, model_number, learner)
                    plt.barh(padding, importances[sorted_idx], align='center')
                    plt.yticks(padding, cols)
                    plt.xlabel("Relative Importance")
                    plt.title(t)
                    plt.tight_layout()
                    fig.savefig(doc)
                except:
                    print('Clf has no feature_importances_ attribute:') 
            
                # results.to_excel(write_to)
    
    print('\n\tBEST MODEL!:\n')
    for key in best:
        print(key, best[key])
    return pd.DataFrame.from_dict(results_dataframe), best