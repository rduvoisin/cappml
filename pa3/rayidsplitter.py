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
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
from model import *


def get_mse(predicted, val_targets):
    return (((predicted - val_targets) ** 2).sum()) / len(predicted)


def build_trainer(trainer, split_list, splits, ModelTrains):
    '''
    Converts the test and split into stored Trainer objects.
    Returns the conjoined X_train trainer object.'''
    for split_X in split_list:
        tag = ''
        y_array = split_X.replace('X', 'y')
        test_X = split_X.replace('train', 'test')
        test_y = y_array.replace('train', 'test')
        if trainer.target in trainer.impute:
            tag += '_for'
            for c in trainer.impute:
                tag += '_{}'.format(c)
        new_Y =  '{}_{}_'.format(y_array, trainer.target) + trainer.name + tag
        new_X =  '{}_{}_'.format(split_X, trainer.target) + trainer.name + tag

        newytrainer = Trainer(new_Y, eval(y_array), trainer.target)
        newytrainer.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newytrainer)
        except:
            pass

        newxtrainer = Trainer(new_X, eval(split_X), trainer.outcome)
        newxtrainer.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newxtrainer)
        except:
            pass

        new_testY =  '{}_{}_'.format(test_y, trainer.target) + trainer.name + tag
        new_testX =  '{}_{}_'.format(test_X, trainer.target) + trainer.name + tag

        newytester = Trainer(new_testY, eval(test_y), trainer.outcome)
        newytester.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newytester)
        except:
            pass

        newxtester = Trainer(new_testX, eval(test_X), trainer.outcome)
        newxtester.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newxtester)
        except:
            pass

        # Join Holdout
        holdoutset = eval(test_y)
        holdoutset.join(eval(test_X))

        newholdout =  '{}_{}_'.format('HOLDOUT', trainer.target) + trainer.name + tag
        holdout = Trainer(newholdout, holdoutset, trainer.outcome)
        holdout.target = trainer.target
        holdout.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(holdout)
        except:
            pass

        # Join Trainer
        trainerset = eval(y_array)
        trainerset.join(eval(split_X))

        newtrainer =  '{}_{}_'.format('TRAIN', trainer.target) + trainer.name + tag
        trainerx = Trainer(newtrainer, trainerset, trainer.outcome, validator = holdout)
        trainerx.target = trainer.target
        trainerx.add_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(holdout)
        except:
            pass

        return trainerx


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    plt.close('all')
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    name = model_name
    plt.title(name)
    plt.savefig(name)
    # plt.show()


def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


def replace_best_model(trainer, best, model_dict, estimator, p, X_train, X_test, y_train, y_test,
                       reg_or_clf, model_number, outcome_variable, 
                       learner, metric):
    '''Sets the best model so far.'''
    if model_dict[metric] > best[metric]:
        try:
            path = 'data/plots/{}'.format(outcome_variable)
            os.mkdir(path)
        except:
            pass
        print('\nMODEL SCORE to beat:', best[metric])
        best['model_dict'] = None
        best[metric] = model_dict[metric]
        best['learner'] = estimator
        best['Classifier'] = model_dict[learner]
        print('\n\tBETTER MODEL!\n')
        print('Model {}.'.format(model_dict['Model']))
        for better_result in model_dict:
            if better_result != 'Model':
                print(better_result, model_dict[better_result])
        best['model_dict'] = model_dict.copy()
        best['PARAMETERS'] = p
        best['model_dict'] = model_dict.copy()
        try:
            importances = estimator.feature_importances_
            sorted_idx = np.argsort(importances)
            padding = np.arange(len(cols)) + 0.5
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10,8))
            t = 'Model {} Imputed {} by {}:\nVariable Importances of {}'.format(
                model_number, outcome_variable, learner, cols)
            doc = '{}_{}_{}_feat_importance.png'.format(filename, model_number, learner)
            plt.barh(padding, importances[sorted_idx], align='center')
            plt.yticks(padding, cols)
            plt.xlabel("Relative Importance")
            plt.title(t)
            plt.tight_layout()
            fig.savefig(doc)
        except:
            print('Clf has no feature_importances_ attribute:') 
            # X_train, X_test, y_train, y_test
        try:
            try:
                probs = estimator.fit(X_train, y_train).predict_proba(X_test)[:,1]
            except:
                pass
            try:
                # print(precision_at_k(y_test,probs,.05))
                k = .05
                x = precision_at_k( y_test, probs, k)
                tag = 'Precision at {}'.format(x, k)
            except:
                tag = ''
            try:
                t = '{}/Model {} Precision-Recall of {} by {}\n{}:'.format(path, model_number, 
                                                                    outcome_variable, learner, tag)
                plot_precision_recall_n(y_test, probs, t)
            except:
                pass
            try:
                t = '{}/Model {} ROC of {} by {}:'.format(path, model_number, outcome_variable, learner)
                plot_roc(t, x_test, probs, estimator, outcome_variable)
            except:
                pass
        trainer.best((outcome_variable, best))



def splitter(trainer, ModelTrains, models_to_run = ['RFR', 'DTR', 'KNNR', 'KNN', 'DT', 'RF','LR','ET','AB','GB'],
                  cols=None, exclude_features=None, testsize=0.20, 
                  results_dataframe=None, regress_only=False,
                  filename=None, threshold=None):
    
    split_list = ['X_train', 'X_test']
    
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
                      'score': [],
                      'precision' : []
                     }
    outcome_variable = trainer.target
    if not filename:
        name = 'data'
        try:
            os.mkdir(name)
        except:
            pass
        try:
            os.mkdir('{}/{}'.format('data', trainer.name))
        except:
            pass
    else:
        try:
            os.mkdir(filename)
        except:
            pass
    filename = filename + '/{}'.format(trainer.name) 
    write_to = filename + '.xlsx'

    # Query which features to use.
    if not cols:
        cols = [c for c in trainer.now.columns.tolist()]
        if isinstance(exclude_features, str):
            exclude_features = [exclude_features]
        if isinstance(exclude_features, list):
            cols = [c for c in cols if c not in exclude_features]
        cols = [c for c in cols if c not in [trainer.target, trainer.outcome]]
    else:
        for c in cols:
            if c not in trainer.now.columns.tolist():
                raise ValueError("{} not included in Trainer object's dataset".format(c))
                return

    # Stage learner storage collector:
    model_number = 0
    best = {'score': float(0), 'model_dict': None, 'Classifier': None, 'Regressor': None}
    if not results_dataframe:
        results_dataframe = results_matrix.copy()
    
    # Split the training data into a training set and a validation set
    dataset = trainer.now.copy()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset[cols], 
                                                                         dataset[trainer.target], 
                                                                         test_size = testsize)
    # Make trainer objects from these.

    newname = ''
    for split_X in split_list:
        tag = ''
        y_array = split_X.replace('X', 'y')
        test_X = split_X.replace('train', 'test')
        test_y = y_array.replace('train', 'test')
        if trainer.target in trainer.impute:
            tag += '_for'
            for c in trainer.impute:
                tag += '_{}'.format(c)
        new_Y =  '{}_{}_'.format(y_array, trainer.target) + trainer.name + tag
        new_X =  '{}_{}_'.format(split_X, trainer.target) + trainer.name + tag

        nypd = pd.DataFrame({trainer.target : eval(y_array)})
        print(nypd.info())
        print(nypd.shape)
        print(nypd.isnull().sum())
        newytrainer = Trainer(new_Y, nypd, trainer.outcome)
        newytrainer.set_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newytrainer)
        except:
            pass

        nxpd = pd.DataFrame(eval(split_X))
        newxtrainer = Trainer(new_X, nxpd, trainer.outcome)
        newxtrainer.set_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newxtrainer)
        except:
            pass

        new_testY =  '{}_{}_'.format(test_y, trainer.target) + trainer.name + tag
        new_testX =  '{}_{}_'.format(test_X, trainer.target) + trainer.name + tag

        typd = pd.DataFrame({trainer.target : eval(test_y)})
        newytester = Trainer(new_testY, typd, trainer.outcome)
        newytester.set_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newytester)
        except:
            pass

        txpd = pd.DataFrame(eval(test_X))
        newxtester = Trainer(new_testX, txpd, trainer.outcome)
        newxtester.set_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(newxtester)
        except:
            pass

        # Join Holdout
        holdoutset = pd.DataFrame({trainer.target : eval(test_y)})
        testx = pd.DataFrame(eval(test_X))
        holdoutset = holdoutset.join(testx)
        newholdout =  '{}_{}_'.format('HOLDOUT', trainer.target) + trainer.name + tag
        print(newholdout, '\n', holdoutset.info(verbose=True))

        holdout = Trainer(newholdout, holdoutset, trainer.outcome)
        holdout.target = trainer.target
        print(holdout.name, '\n', holdout.now.info(verbose=True))
        holdout.set_parent(trainer, ModelTrains)
        try:
            ModelTrains.add(holdout)
        except:
            pass

        # Join Trainer
        trainerset = pd.DataFrame({trainer.target : eval(y_array)})
        trainx = pd.DataFrame(eval(split_X))
        trainerset = trainerset.join(pd.DataFrame(trainx))
        newtrainer =  '{}_{}_'.format('TRAIN', trainer.target) + trainer.name + tag

        print(newtrainer, '\n', trainerset.info(verbose=True))
        trainerx = Trainer(newtrainer, trainerset, trainer.outcome, validator = holdout)
        trainerx.target = trainer.target
        print(trainerx.name, '\n', trainerx.now.info(verbose=True))
        trainerx.set_parent(trainer, ModelTrains)
        newname = newtrainer
        try:
            ModelTrains.add(trainerx)
        except:
            pass

    ModelTrains.show()
    XTrain = ModelTrains.get(newname)
    XVal = XTrain.validator
    
    # Make Imputer Children Trainers:
    # XTrain with medians for any variable that is not the target of imputation.
    # for impute_var in XTrain.impute:
    imputation_stats_and_methods = {}
    XTOR = XTrain.now.copy()
    while (XTrain.impute > len(imputation_stats_and_methods.keys())):
        for imputer_var in XTrain.impute:
            XT = XTOR.copy()
            XMOD = XTrain.now.copy()
            XTrain.target = imputer_var
            XTrain.impute = XTrain.impute.pop(0)
            imputer, params, cols = splitter(XTrain, ModelTrains, 
                    models_to_run = ['RFR', 'DTR', 'KNNR', 'KNN', 'DT', 'RF','LR','ET','AB','GB'],
                    cols=None, exclude_features=None, testsize=0.20, 
                    results_dataframe=None, regress_only=False,
                    filename=filename)
            imputer.set_params(**params)
            imputation_stats_and_methods[imputer_var] = (imputer, cols)
            XMOD[imputer_var] = XT[imputer_var]
            # Split data into cases that reported the imputation feature versus those that didn't
            have_it = XMOD[XMOD[imputer_var].isnull() == False]
            print(have_it.shape)
            dont_have_it = XMOD[XMOD[imputer_var].isnull() == True]
            dont_have_it.isnull().sum()
            imputer.fit(have_it[cols], have_it[imputer_var])
            new_imputations = imputer.predict(dont_have_it[cols])
            dont_have_it[imputer_var] = new_imputations
            
            combined = have_it.append(dont_have_it)
            XMOD[imputer_var] = combined[imputer_var]
            # Updatelogs
            checklog = imputer_var + '_log'
            if checklog in XMOD.columns.tolist():
                XMOD[checklog] = XMOD[feature].apply(lambda x: np.log(x) if x > 0 else np.log(x + 1))
            XTrain.set_data(XMOD.copy())
            XTrain.imputed = imputer_var

    XTrain.target = outcome_variable
    XMOD = XTrain.now.copy()

    # Now run thru the loop to get the best clfs.
    print('regress_only?', regress_only)
    model_number = 0
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        model_dict = {}
        print(models_to_run[index])
        learner = models_to_run[index]
        model_dict['learner'] = learner
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            model_number += 1
            for k in results_dataframe.keys():
                results_dataframe[k].append(np.nan)
                print(k, results_dataframe[k])
                model_dict[k] =  np.nan
            try:
                clf.set_params(**p)
                print(clf)
            except:
                continue
            
            
            model_dict['Model'] = model_number
            model_dict['Y_outcome'] = outcome_variable
            model_dict['Training_set'] = dataset
            model_dict['Test_size'] = testsize 
            model_dict['Classifier'] = clf       

            # Modify any remaining missing variables to their medians and Fit.
            try:
                estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                              strategy="median",
                                              axis=0)),
                                    (learner, clf)])
                score = cross_val_score(estimator, X_train, y_train).mean()
            except:
                score = np.nan
                estimator = clf

            
            #Fit the model to the training inputs and training targets
            model_dict['Predictors'] = cols
            model_dict['score'] = score

            estimator.fit(X_train, y_train)

            #Predict the output on the validation
            predicted = estimator.predict(X_test)
            mse = get_mse(predicted, y_test)
            model_dict['metric_score'] = mse
                
            try:
                scores = cross_val_score(estimator, X_train, y_test).mean()
                model_dict['cross_val_metric'] = scores
                # model_dict['score'] = scores
                # score = scores
            except:
                model_dict['cross_val_metric'] = np.nan
            for k in results_dataframe:
                print(k, len(results_dataframe[k]))
            for k in model_dict:
                print(k, model_dict[k])
            #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
            #print threshold
            print(precision_at_k(y_test,y_pred_probs,.05))
            # plot_precision_recall_n(y_test,y_pred_probs,clf)

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
            results.to_excel(write_to)
            
            # if model_dict['score'] > best['score']:
            #     print('\nMODEL SCORE to beat:', best['score'])
            #     best['model_dict'] = None
            #     best['score'] = model_dict['score']
            #     # best['learner'] = model_dict[learner]
            #     best['Classifier'] = model_dict['Classifier']
            #     print('\n\tBETTER MODEL!\n')
            #     print('Model {}.'.format(model_dict['Model']))
            #     for better_result in model_dict:
            #         if better_result != 'Model':
            #             print(better_result, model_dict[better_result])
            #     best['model_dict'] = model_dict.copy()
            #     best['PARAMETERS'] = p
            #     best['model_dict'] = model_dict.copy()
            #     try:
            #         importances = estimator.feature_importances_
            #         sorted_idx = np.argsort(importances)
            #         padding = np.arange(len(cols)) + 0.5
            #         plt.close('all')
            #         fig, ax = plt.subplots(figsize=(10,8))
            #         t = 'Model {} Imputed {} by {}:\nVariable Importances of {}'.format(
            #             model_number, outcome_variable, learner, cols)
            #         doc = '{}_{}_{}_feat_importance.png'.format(filename, model_number, learner)
            #         plt.barh(padding, importances[sorted_idx], align='center')
            #         plt.yticks(padding, cols)
            #         plt.xlabel("Relative Importance")
            #         plt.title(t)
            #         plt.tight_layout()
            #         fig.savefig(doc)
            #     except:
            #         print('Clf has no feature_importances_ attribute:') 
            if model_dict['cross_val_metric'] > model_dict['cross_val_metric']:
                print('\n****\tMODEL UPDATE\t*****:')
                best = replace_best_model(trainer, best, model_dict, estimator, 
                                          parameters=p, X_train, X_test, y_train, y_test,
                                          model_number, outcome_variable, 
                                          learner, 'cross_val_metric', filename=filename)
            
    print('\n\tBEST MODEL!:\n')
    for key in best:
        print(key, best[key])
    return best['learner'], best['PARAMETERS'], cols





