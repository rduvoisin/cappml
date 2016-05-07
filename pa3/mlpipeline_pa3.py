#import libraries for pipeline
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import matplotlib.cm as cm
import matplotlib.patches as patches
import math
import time
import json
import requests
import hide_code
import notebook
import re
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from splitter import *
from model import *
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.ensemble import RandomForestRegressor as RandomForestR
from sklearn.tree import DecisionTreeRegressor as DecisionTreeR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.ensemble import RandomForestClassifier as RandomForestC
from sklearn.tree import DecisionTreeClassifier as DecisionTreeC
from sklearn.ensemble import GradientBoostingClassifier as GradientBoosting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from scipy.stats.mstats import mquantiles
# get_ipython().magic('matplotlib inline')



# Plotting functions:
def plot_roc(name, dataset, probs, clf, outcome):
    fpr, tpr, thresholds = roc_curve(dataset[outcome], probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(name)


def plot_confusion_matrix(cm, title='Confusion matrix', doc='outcome_confusion_matrix', cmap=plt.cm.Blues):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(doc)




# Determine which features require preprocessing.
def list_features_wmissing(dataset):
    '''
    Returns all features that have missing values:
        - a list of just those features.'''
    print('Summary Statistics on Full Data set:\n{}'.format(dataset.describe(include='all').round(2)))
    has_null = pd.DataFrame({'Total_missings' : dataset.isnull().sum()})
    has_null[(has_null.Total_missings >0)].index.tolist()
    print('\n\n{} Features containing missing values: {}\n'
          .format(len(has_null[(has_null.Total_missings >0)].index.tolist()),
           has_null[(has_null.Total_missings >0)].index.tolist()))
    print(has_null.ix[2:,:])
    return has_null[(has_null.Total_missings >0)].index.tolist()



# Feed this list as input to a general function that
# trains a model of missings imputations for each feature.
def get_correlates_dict(dataset, feature_wnull, not_same=True, output_variable=None):
    correlated_predictionary = {}
    for target in feature_wnull:
        if target not in output_variable:
            try:
                corrMatrix = pd.DataFrame({'Correlation': dataset.corr().ix[:,target].sort_values()})
                corrAbsMatrix = corrMatrix.copy()
                corrAbsMatrix['Absolute'] = corrAbsMatrix.Correlation.apply(lambda x : abs(x))
                corrAbsMatrix.sort_values('Absolute', inplace=True)
                print('\nCorrelation Matrix with Respect to {}:\n'.format(target),
                      corrAbsMatrix)
                iteration = -3
                best_correlates = corrAbsMatrix.index.tolist()                                  [iteration : len(corrAbsMatrix.index.tolist())]
                if not_same:
                    # Remove any similarly named correlates that
                    # likely to be transformations of the same target variable.
                    exclude_by_name = []
                    for c in corrAbsMatrix.index.tolist():
                        similar = re.search(target, c)
                        if similar:
                            exclude_by_name.append(c)
                    best_correlates = [choice for choice in best_correlates
                                       if choice not in exclude_by_name]
                    if output_variable:
                        exclude_by_name.append(output_variable)
                    iteration = -3
                    while len(best_correlates) < 3:
                        iteration -= 1
                        best_correlates = corrAbsMatrix.index.tolist()                                          [iteration : len(corrAbsMatrix.index.tolist())]
                        best_correlates = [choice for choice in best_correlates
                                           if choice not in exclude_by_name]
                print('\nTOP 3 Correlates with {}: (EXCLUDES {})\n{}\n\n{}'.
                      format(target, exclude_by_name, corrAbsMatrix.tail(abs(iteration)), best_correlates))
                correlated_predictionary[target] = best_correlates
            except:
                print('get_correlates_dict ERROR: {}'.format(target))
    return correlated_predictionary


# Feed this dictionary into a function that
# trains a model of missings imputations
def get_encoded_features_list(dataset, discrete_threshold=None, excepting=None, only=None):
    '''Returns a list of discrete variables that
       have numerical values signifying missing.
       (i.e. 96=DK, 98=REFUSED)'''
    contain_missing = []
    print('\nINSPECT DISCRETE FEATURES FOR ENCODED MISSISINGS:\n')
    if not only:
        only = dataset.columns.tolist()
    only = [feature for feature in only if feature not in excepting]
    for col in only:
        if discrete_threshold:
            if len(dataset[col].unique()) < discrete_threshold:
                print('{} unique values:\n{}'.format(col, dataset[col].unique()))
                contain_missing += [col]
        else:
            contain_missing += [col]
            print('{} unique values:\n{}'.format(col, dataset[col].unique()))
    print('\nCheck for coded missings on these features: \n{}'.format(contain_missing))
    return contain_missing


def decode_extended_to_nan(dataset, contain_missing, to_replace=None, values=None):
    '''Inputs: Dataframe and list of features to encode.
       Returns a dataframe with specified encoded missings
       replaced to NAN.'''
    if (len(to_replace) == len(values)) & (isinstance(to_replace, list)):
        dataset[contain_missing] = dataset[contain_missing].replace(to_replace, values)
        new_missings = pd.DataFrame({'Total_missing' :
                                    dataset.isnull().sum().round(2)})
        print('\nEncode obs with NAN in any of the extended missings columns: \n{}\
            \nAfter Encoding Extended Missings\n{},\n\nDrop obs with extended missings'.
              format(contain_missing, new_missings))
        return dataset, new_missings
    else:
        raise ValueError("Provide symmetrical number lists to 'to_replace' and 'values'.")


def drop_obs_w_anynan(dataset, features_list):
    '''Inputs: Dataframe and list of features to encode.
       Returns a dataset removed any obs missing
       on the specified by the features_list.'''
    excludes = [feature for feature in dataset.columns.tolist() if feature not in features_list]
    if excludes:
        print('\nDropping all row-wise missing values, EXCEPT for features:\n{}'
             .format(excludes))
    else:
        print('\nDropping all row-wise missing values for features:\n{}'
              .format(features_list))
    dataset.dropna(axis='index', how='any', inplace=True, subset=[features_list])
    print('\nAfter Dropping Extended Missings\n{}\n'
          .format(pd.DataFrame(
                 {'Total_missings' : dataset.isnull().sum()})))
    return dataset


def decode_and_drop_missings(raw_train, decodings_dict, except_threshold=None, encode_except=None, outcome_variable=None):
    '''Inputs:
        - Dataframe
        - Decoding variables to values dictionary
        - Optional discrete_threshold (i.e. 1000 unique values)
       Returns '''
    encoded_features = get_encoded_features_list(raw_train,
                                                 except_threshold,
                                                 excepting=encode_except)

    for i in range(len(decodings_dict.keys())):
        if not decodings_dict[i]['on']:
            decodings_dict[i]['on'] = encoded_features
        else:
            encoded_features.extend(decodings_dict[i]['on'])
        raw_train, imputation_candidates =         decode_extended_to_nan(raw_train, decodings_dict[i]['on'],
                               to_replace = decodings_dict[i]['to_replace'],
                               values = decodings_dict[i]['with_replace'])

    # Derive binary missing indicator variables
    dropped_train = raw_train.copy()
    derived_train = raw_train.copy()
    inspect_missing_list = []
    for feature in imputation_candidates[imputation_candidates.Total_missing > 0].index:
        if feature not in outcome_variable:
            inspect_missing_list += [feature]
            is_missing_var = feature + '_missing'
            derived_train[is_missing_var] = derived_train[feature].isnull().map({True : 1, False : 0})

    # Drop all missings
    train_missing = dropped_train.copy()

    if outcome_variable:
        dropping_columns = [c for c in derived_train.index.tolist() if c not in outcome_variable]
        derived_train.dropna(how='any', axis=1, subset=[dropping_columns])
    else:
        derived_train.dropna(how='any', axis=1)
    dropped_train =  drop_obs_w_anynan(dropped_train, encoded_features).copy()
    return dropped_train, derived_train, train_missing, inspect_missing_list



# Transform Data
def gen_transform_data(dataset, transform_dict, transformations=None):
    '''Inputs:
        - Dataset
        - dictionary of features to transform with
            respective transformation function.'''
    if not transformations:
        transformations = {'log': 'np.log'}
    for feature in transform_dict.keys():
        fx = transform_dict[feature]
        plus = 0
        if fx in transformations:
            new_feature = feature + '_' + fx
            print('\nTransforming {} by way of {} = {}'
                  .format(feature, fx, new_feature))
            if fx == 'log':
                plus = 1
            dataset[new_feature] = dataset[feature].apply(lambda x : eval(transformations[fx])(x + plus))
        else:
            raise ValueError("Provide valid transformation function. {} is invalid.".format(fx))
    return dataset


def get_mse(predicted, val_targets):
    return (((predicted - val_targets) ** 2).sum()) / len(predicted)


def replace_best_model(best, model_dict, clf, parameters, X_train, 
                       X_val, reg_or_clf, model_number, outcome_variable, 
                       learner, metric, filename='best'):
    print('\nMODEL SCORE ({}) to beat:'.format(metric), best[metric])
    best['model_dict'] = None
    best[metric] = model_dict[metric]
    best['Regressor'] = model_dict['Regressor']
    best['Classifier'] = model_dict['Classifier']
    print('\n\tBETTER MODEL!\n')
    print('Model {}.'.format(model_dict['Model']))
    for better_result in model_dict:
        if better_result != 'Model':
            print(better_result, model_dict[better_result])
    best['model_dict'] = model_dict.copy()
    best['PARAMETERS'] = parameters.copy()
    best['model_dict'] = model_dict.copy()
    try:
        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)
        padding = np.arange(len(cols)) + 0.5
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10,8))
        t = 'Model {} Imputed {} by {}:\nVariable Importances of {}'.format(model_number, 
                                                                            outcome_variable, 
                                                                            learner, cols)
        doc = '{}_{}_{}_feat_importance.png'.format(filename, model_number, learner)
        plt.barh(padding, importances[sorted_idx], align='center')
        plt.yticks(padding, cols)
        plt.xlabel("Relative Importance")
        plt.title(t)
        plt.tight_layout()
        fig.savefig(doc)
    except:
        print('CLF importances not available')  
    if reg_or_clf=="C":
        t = 'Model {} of {} by {}:'.format(model_number, outcome_variable, learner)
        probs = clf.fit(X_train[cols], X_train[outcome_variable]).predict_proba(X_val[cols])
        
        plot_roc(t, X_val, probs, clf, outcome_variable)
    return best.copy()




def get_impute_model(to_impute, to_avg, outcome, train_impute, train_missing_transformed):
    '''INPUTS: 
        - Features to impute by modelling,
        - Features to impute by average metric
        - Outcome variables to ignore
        - imputation training set to return
        - a version of the training set before imputation
        
        RETURNS:IMPUTATIONS (a dictionary of features to imputation methods)
        - train_impute (a dataframe with imputed values)
        '''
    drop_after = []
    IMPUTATIONS = {}
    print('\nSTART TRAIN_IMPUTE\n', train_impute.shape)
    for feature in to_impute:
        # Specify predictive features (determined from training set)
        if feature == 'MonthlyIncome':
            cols = ['NumberOfDependents', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']
        elif feature == 'NumberOfDependents':
            cols = ['MonthlyIncome_log', 'NumberRealEstateLoansOrLines', 'age'] 

        train_impute[feature] = train_missing_transformed[feature].copy()   
        for other_feature in to_impute:
            if other_feature != feature:
                mean_other = other_feature + '_median'
                drop_after.append(mean_other)
                train_impute[mean_other] = \
                    train_impute[other_feature].fillna(train_impute[other_feature].median())
                train_impute[other_feature] = \
                    train_impute[other_feature].fillna(train_impute[other_feature].median())
        for dismiss_feature in to_avg:
            train_impute[dismiss_feature] = \
                train_impute[dismiss_feature].fillna(train_impute[dismiss_feature].median())
            IMPUTATIONS[dismiss_feature] = 'MEDIAN'
        print(train_impute.isnull().sum())

        # Store results of the imputation model on the test/training set.
        file = 'test_impute_' + feature
        file = 'train_impute_' + feature

        # #Split data into people who reported the imputation feature versus those that didn't
        # have_it = train_impute[train_impute[feature].isnull()==False].copy()
        # dont_have_it = train_impute[train_impute[feature].isnull()==True].copy()

        # Deal with missings. Update model_builder for imputing.
        model_builder_impute = model_builder.copy()
        model_builder_impute['trainees'] = ['train_trainsformed']
        print('MODEL BUILDER IMPUTE (during get_impute_model)\n')
        print(model_builder.keys())
        for k in model_builder:
            if k == 'trainees':
                print('\n{} ({}), shaped {} : \n{}'.format(k, len(model_builder[k]), 
                    model_builder[k][0].shape, model_builder[k][0].isnull().sum()))
            else:
                print('\n{} : \n{}'.format(k, model_builder[k]))
        results_df, best_model = split_and_run(model_builder_impute, feature, 
                                               reg_or_clf='R', impute_dictionary=False,
                                               filename=file)
        
        print('\nCONVERGED! Imputing {} with the best fit model:\n{}'.format(feature, best_model))
        print(train_impute.isnull().sum())
        print('\ntrain_impute\n', train_impute.shape)
        
        # Compile selected classifier:
        if best_model['Regressor']:
            print('REGRESSED!\t\n', best_model['Regressor'])
            clf = eval(best_model['Regressor'])()
        if len(best_model['PARAMETERS'].keys()) > 0:
            for param in best_model['PARAMETERS']:
                clf.set_params(**{param:best_model['PARAMETERS'][param]})
        IMPUTATIONS[feature] = clf
        print(train_impute.isnull().sum())
        
        #Split data into people who reported the imputation feature versus those that didn't
        have_it = train_impute[train_impute[feature].isnull()==False].copy()
        dont_have_it = train_impute[train_impute[feature].isnull()==True].copy()
        
        # Impute with the best model for features with missings (Monthly Income and Number of Dependents).
        imputer = clf 
        imputer.fit(have_it[cols].as_matrix(), have_it[feature].as_matrix())
        new_imputations = imputer.predict(dont_have_it[cols].as_matrix())
        dont_have_it[feature] = new_imputations
        combined = have_it.append(dont_have_it)
        train_impute[feature] = combined[feature].copy()
        checklog = feature + '_log'
        print('\nchecklog', checklog)
        print(train_impute.columns.tolist())
        print(train_impute.isnull().sum())
        if checklog in train_impute.columns.tolist():
            train_impute[checklog] = \
            train_impute[feature].apply(lambda x: np.log(x) if x > 0 else np.log(x + 1))
            print('\ntrain_impute[checklog]', train_impute[checklog].head())
        else:
            print('NO LOG!')
    print('\ntrain_impute\n', train_impute.shape)
    train_impute.drop(drop_after, axis=1, inplace=True)
    print('\ntrain_impute\n', train_impute.shape)
    return IMPUTATIONS, train_impute
    
# IMPUTATIONS_METHODS, DATASET = get_impute_model(to_impute, to_avg, outcome, 
#                                                 train_impute, train_missing_transformed)

# def split_and_run(classifier_dictionary, outcome_variable, 
#                   reg_or_clf='R', 
#                   cols=None, test_size=0.20, results_dataframe=None, 
#                   impute_dictionary=False, to_impute=False,
#                   filename='output', 
#                   update_for_imputation=None, 
#                   original_train_bf_imputation= None):
    
#     results_matrix = {'Model': [], 
#                       'Training_set': [], 
#                       'Y_outcome' : [],
#                       'Test_size': [],
#                       'Classifier': [],
#                       'Regressor': [],
#                       'Predictors': [],
#                       'n_estimators': [],
#                       'max_depth': [],
#                       'min_samples_split': [],
#                       'metric': [],
#                       'metric_score': [],
#                       'r2_metric': [],
#                       'r2_score' : [],
#                       'cross_val_metric': [],
#                       'score': [],
#                       'precision' :[], 
#                       'recall': [], 
#                       'roc_auc': [],
#                       'f1_score': []
#                      }
#     write_to = filename + '.xlsx'
        
#     # Loop through models in classifier_dictionary
#     if update_for_imputation:
#         print('\n*****SPLIT AND RUN CALLED FOR IMPUTATION******\n')
#     else:
#         print('\nSPLIT AND RUN CALLED FOR CLASSIFICATION\n')
#     model_number = 0
#     best = {'score': float(0), 'model_dict': None, 'Classifier': None, 'Regressor': None, 'roc_auc': float(0)}
#     if not results_dataframe:
#         results_dataframe = results_matrix.copy()
#     for dataset in range(len(classifier_dictionary['trainees'])):
#         # Split the training data into a training set and a validation set
#         if not cols:
#             if outcome_variable in classifier_dictionary['versions'][dataset]:
#                 cols = classifier_dictionary['versions'][dataset][outcome_variable]
#             else:
#                 print('Outcome {} not in dataset {}'.format(outcome_variable, dataset))
#                 continue
#         for testsize in classifier_dictionary['test_sizes']:
#             print('JUST B/F SPLIT\n',classifier_dictionary['trainees'][dataset].shape,
#                 '\n',classifier_dictionary['trainees'][dataset].isnull().sum())
#             X_train, X_val = cross_validation.train_test_split(classifier_dictionary['trainees'][dataset],
#                                                                test_size = testsize)
#             X_train.isnull().sum()
#             X_train_original = X_train.copy()
#             if impute_dictionary:
#                 to_avg = [c for c in X_train.columns.tolist() if c not in to_impute]
#                 outcome = outcome_variable
#                 to_avg = [c for c in to_avg if c not in outcome]
#                 classifier_dictionary_for_impute = classifier_dictionary.copy()
#                 if update_for_imputation:
#                     classifier_dictionary_for_impute['trainees'] = update_for_imputation
#                     print('classifier_dictionary_for_impute', classifier_dictionary_for_impute)
#                 original = X_train_original.copy()
#                 train_impute = X_train.copy()
#                 # Prep Imputations of Training Set.
#                 # get_impute_model(to_impute, to_avg, outcome, train_impute, train_missing_transformed, model_builder)
#                 # pass a list of features to impute,
#                 # pass a subtraining dataset to use.
#                 # pass a raw version to pull missing data from
#                 # pass a classifier dictionary to use in the split call
#                 UPDATED_IMPUTATIONS, X_train = get_impute_model(to_impute, 
#                                                                 to_avg, 
#                                                                 outcome, 
#                                                                 X_train, 
#                                                                 original, 
#                                                                 classifier_dictionary_for_impute)
#                 print('\nXTRAIN IMPUTATION\n', X_train.isnull().sum())
#                 print('\nUPDATED_IMPUTATIONS\n', UPDATED_IMPUTATIONS)
#             print('\nX_train.shape\n', X_train.shape)
#             print('\nreg_or_clf\n', reg_or_clf)
#             # Build the ML Regressor/Classifier
#             if reg_or_clf == 'C':
#                 trainer_arg = 'Classifier'
#                 trainer_counter_arg = 'Regressor'
#             elif reg_or_clf == 'R':
#                 trainer_arg = 'Regressor'
#                 trainer_counter_arg = 'Classifier'
#             for learner in classifier_dictionary[reg_or_clf]:
#                 model_dict = {}
#                 print('\n\t\t\t\t\tLearner: {}\n'.format(learner))
#                 for stats_key in classifier_dictionary[reg_or_clf][learner].keys():  
#                     if stats_key not in results_dataframe:
#                         results_dataframe[stats_key] = []
#                         for i in range(model_number - len(results_dataframe[stats_key])):
#                             results_dataframe[stats_key].append(np.nan)
#                     for stat_value in classifier_dictionary[reg_or_clf][learner][stats_key]: 
#                         model_dict[stats_key] = stat_value
                        
#                         model_number += 1

#                         parameters = {}
#                         for model_key in model_dict:
#                             if model_key in classifier_dictionary[reg_or_clf][learner]:
#                                 if model_dict[model_key]!="NA":
#                                     parameters[model_key] = model_dict[model_key]

#                         model_dict['Model'] = model_number
#                         model_dict['Y_outcome'] = outcome_variable
#                         model_dict['Training_set'] = dataset
#                         model_dict['Test_size'] = testsize        
#                         model_dict[trainer_arg] = learner
#                         model_dict[trainer_counter_arg] = np.nan
                        

#                         clf = eval(learner)()
#                         clf_args = clf.get_params()
#                         for param in parameters:
#                             clf.set_params(**{param:parameters[param]})
                            
#                         #Fit the model to the training inputs and training targets
#                         model_dict['Predictors'] = cols
#                         clf.fit(X_train[cols].as_matrix(), X_train[outcome_variable].as_matrix())

#                         #Predict the output on the validation
#                         predicted = clf.predict(X_val[cols].as_matrix())
                        

#                         X_val[outcome_variable + '_predicted'] = predicted
#                         mse = get_mse(predicted, X_val[outcome_variable].as_matrix())
                        
#                         model_dict['metric'] = 'mse'
                         
#                         model_dict['metric_score'] = mse
                            
#                         score = np.nan
                        
#                         try:
#                             score = clf.score(X_val[cols], X_val[outcome_variable], sample_weight=None)
#                             model_dict['score'] = score
#                             score = model_dict['score']
#                         except:
#                             print('Clf gave no score: {}.\nSCORE is :{}'.format(clf, score))
#                         try:
#                             scores = cross_val_score(clf, X_val[cols], X_val[outcome_variable])
#                             model_dict['cross_val_metric'] = 'cross_val_score'
#                             model_dict['score'] = scores
#                             score = scores
#                         except:
#                             model_dict['cross_val_metric'] = 'NA'
#                         model_dict['score'] = score
                        
#                         r2_scoring = np.nan
#                         try:
#                             r2_scoring = r2_score(y_true = X_val[outcome_variable].as_matrix(), 
#                                                   y_pred = X_val[outcome_variable + '_predicted'].as_matrix())
#                             model_dict['r2_metric'] = 'r2'
#                         except:
#                             model_dict['r2_metric'] = 'NA'
#                             err_code = np.nan 
#                         model_dict['r2_score'] = r2_scoring
                            
#                         # Store Classification Metrics
                        
#                         f1_scoring, precision, recall, roc_auc = np.nan, np.nan, np.nan, np.nan
#                         if reg_or_clf =='C':
#                             class_rpt = classification_report(X_val[outcome_variable].as_matrix(), 
#                                                               X_val[outcome_variable + '_predicted'].as_matrix())
#                             print('CLASSIFICATION REPORT:\n', class_rpt)
#                             try:
#                                 precision = precision_score(y_true = X_val[outcome_variable].as_matrix(), 
#                                                             y_pred = 
#                                                             X_val[outcome_variable + '_predicted'].as_matrix())
#                             except:
#                                 err_code = np.nan 
#                             try:
#                                 recall = recall_score(y_true = X_val[outcome_variable].as_matrix(), 
#                                                       y_pred = 
#                                                       X_val[outcome_variable + '_predicted'].as_matrix())
#                             except:
#                                 err_code = np.nan
#                             try:
#                                 array_of_classprobs = predict_proba(X_val[cols])
#                                 roc_auc = roc_auc_score(y_true = X_val[outcome_variable].as_matrix(), 
#                                                         y_score = array_of_classprobs)
#                             except:
#                                 err_code = np.nan 
#                             try:
#                                 f1_scoring = f1_score(y_true = X_val[outcome_variable].as_matrix(), 
#                                                       y_pred = 
#                                                       X_val[outcome_variable + '_predicted'].as_matrix())
#                             except:
#                                 err_code = np.nan 
                                
#                         model_dict['precision'] = precision                        
#                         model_dict['recall'] = recall                          
#                         model_dict['roc_auc'] = roc_auc
#                         model_dict['f1_score'] = f1_scoring 
                            
#                         # Sweep up any new or unused results keys into the results dictionary.
#                         for element in model_dict:
#                             if element not in results_dataframe:
#                                 results_dataframe[element] = []
#                                 for i in range(model_number - len(results_dataframe[element])):
#                                     results_dataframe[element].append(np.nan)
#                                 results_dataframe[element].append(model_dict[element])

#                         for unused_k in results_dataframe:
#                             if unused_k not in model_dict:
#                                 for i in range(model_number - len(results_dataframe[unused_k])):
#                                     results_dataframe[unused_k].append(np.nan)
#                             else:
#                                 results_dataframe[unused_k].append(model_dict[unused_k])
                        
#                         if model_dict['score'] > best['score']:
#                             print('\n****\tMODEL UPDATE\t*****:')
#                             best = replace_best_model(best, model_dict, clf, 
#                                                       parameters, X_train.copy(), X_val.copy(),
#                                                       reg_or_clf, model_number, outcome_variable, 
#                                                       learner, 'score', filename=filename)

#                         # Export to excel for review.
#                         results = pd.DataFrame.from_dict(results_dataframe)
#                         results.to_excel(write_to)
#                         del X_val[outcome_variable + '_predicted']
#     print('\n\tBEST MODEL!:\n')
#     for key in best:
#         print(key, best[key])
#     return pd.DataFrame.from_dict(results_dataframe), best





