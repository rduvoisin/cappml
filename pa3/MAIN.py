# MAIN
#Things to try: Logistic Regression, Random Forest, Gradient Boosting
from __future__ import division
import os
import sys
# import mlpipeline_pa3
from mlpipeline_pa3 import *
from rayidsplitter import *
# %matplotlib inline
# Read in datasets for modelling under cross validation.

datadir = 'data'
Delinquency = ModelTrains(filedir=datadir, title='Delinquency')
DATA = [(t.name, t) for t in Delinquency.trainers]
# [{'TRAIN_PARENT': <model.Trainer at 0x7f6ef331c160>},
#  {'HOLDOUT': <model.Trainer at 0x7f6ef331c6d8>},
#  {'TRAIN': <model.Trainer at 0x7f6ef331c908>},
#  {'TEST': <model.Trainer at 0x7f6ef331cb38>}]
T = Delinquency.get(DATA[2][0])

# Explore Plots:
# Explore the data,each feature individualy, so as not to
# miss key characteristics like outliers and encoded missing values,
# mascarading as true values:
inspection = [T.outcome, 'MonthlyIncome', 'DebtRatio', 'age']
dataplotdir = datadir + '/{}'.format('plots')
inspect_zeros(T, filedir=dataplotdir, inspect=inspection)
dataplotdirZERO = dataplotdir + '/{}/zeros'.format(T.name) 

inspect=['age','MonthlyIncome','DebtRatio',
                  'RevolvingUtilizationOfUnsecuredLines',
                  'SeriousDlqin2yrs']
inspect_pairplot(T, filedir=dataplotdir, inspect=inspect)

# Deal with missing values.
# Collect candidates for imputations of missings. 
feature_wnull = list_features_wmissing(T.now.copy())
<<<<<<< HEAD
print('\n', T.name, 'Correlates\n', T.now.columns)
correlated_features = get_correlates_dict(T, feature_wnull, output_variable=T.outcome)
=======
<<<<<<< HEAD
print('\n', T.name, 'Correlates\n', T.now.columns)
correlated_features = get_correlates_dict(T, feature_wnull, output_variable=T.outcome)
=======
correlated_features = get_correlates_dict(T.now.copy(), feature_wnull, output_variable=T.outcome)
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
>>>>>>> 316feea342fec16110240f43f12cc657cc8b9a3d
decodings_dict = {0 :{'on': None, 'to_replace' : [96, 98],
                  'with_replace' : [np.nan, np.nan]},
                  1:{'on': ['age'],
                  'to_replace' : [0],
                  'with_replace' : [np.nan]}}

# Set codified missing values to missing. 
# Prep training data to trace missing values (row wise and column-wise drops).
# train_alldropped, \
# derived_train, \
# train_missing, \
nothese = decodings_dict[1]['on'][:]
nothese.append(T.outcome)
print(nothese)
imputation_candidates = \
decode_and_drop_missings(T, Delinquency,
                        decodings_dict, 
                        encode_except=nothese,
                        outcome_variable=[T.outcome])

print('\nSummarized Data After Removing Cases with Missing Values:\n')
inspect_correlations(Delinquency)
<<<<<<< HEAD

# Save tracer datasets as Trainer objects.(name, dataframe, outcome_name, validator = None, ModelTrainIndex = None))
transform_features_dict = {'RevolvingUtilizationOfUnsecuredLines' : 'log',
                           'MonthlyIncome' : 'log'}
coldrop = Delinquency.get('COL_DROP')
rowdrop = Delinquency.get('ROW_DROP')
gen_transform_data(Delinquency.get('ROW_DROP'), Delinquency, transform_features_dict)
gen_transform_data(Delinquency.get('COL_DROP'), Delinquency, transform_features_dict)


dropped_columns_correlates = \
get_correlates_dict(Delinquency.get('COL_DROP_log'),
                    Delinquency.get('COL_DROP_log').now.columns,
                    not_same=True, output_variable=Delinquency.get('COL_DROP_log').outcome)

dropped_rows_correlates = \
get_correlates_dict(Delinquency.get('ROW_DROP_log'),
                    Delinquency.get('ROW_DROP_log').now.columns,
                    not_same=True, output_variable=Delinquency.get('ROW_DROP_log').outcome)

Delinquency.show()

last2trainers = [Delinquency.get(len(Delinquency.trainers) - 2)]
last2trainers.append(Delinquency.get(len(Delinquency.trainers) - 1))

# Recall imputation candidates (features with missing data before drop)
print('CONSIDER THESE IMPUTATION CANDIDATES AFTER LOOKING AT COEFFICIENTS OF FIRST APPROXIMATION:\n',
imputation_candidates)

=======
<<<<<<< HEAD

# Save tracer datasets as Trainer objects.(name, dataframe, outcome_name, validator = None, ModelTrainIndex = None))
transform_features_dict = {'RevolvingUtilizationOfUnsecuredLines' : 'log',
                           'MonthlyIncome' : 'log'}
coldrop = Delinquency.get('COL_DROP')
rowdrop = Delinquency.get('ROW_DROP')
gen_transform_data(Delinquency.get('ROW_DROP'), Delinquency, transform_features_dict)
gen_transform_data(Delinquency.get('COL_DROP'), Delinquency, transform_features_dict)


dropped_columns_correlates = \
get_correlates_dict(Delinquency.get('COL_DROP_log'),
                    Delinquency.get('COL_DROP_log').now.columns,
                    not_same=True, output_variable=Delinquency.get('COL_DROP_log').outcome)
=======
# Trow = Delinquency.get('ROW_DROP')
# Trow.now.describe(include='all').round(2)

# Save tracer datasets as Trainer objects.(name, dataframe, outcome_name, validator = None, ModelTrainIndex = None))
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593

dropped_rows_correlates = \
get_correlates_dict(Delinquency.get('ROW_DROP_log'),
                    Delinquency.get('ROW_DROP_log').now.columns,
                    not_same=True, output_variable=Delinquency.get('ROW_DROP_log').outcome)

Delinquency.show()

last2trainers = [Delinquency.get(len(Delinquency.trainers) - 2)]
last2trainers.append(Delinquency.get(len(Delinquency.trainers) - 1))

# Recall imputation candidates (features with missing data before drop)
print('CONSIDER THESE IMPUTATION CANDIDATES AFTER LOOKING AT COEFFICIENTS OF FIRST APPROXIMATION:\n',
imputation_candidates)

>>>>>>> 316feea342fec16110240f43f12cc657cc8b9a3d
# Recall train_derived_transformed (contains missings binaries)
# train_derived_transformed.isnull().sum()
for t in last2trainers:
    print(t.id, t.name, t.shape, t.parent.name, t)


# Decide which dataset to use for imputation:
# Corrected TRAIN plus any Transformed features:
gen_transform_data(Delinquency.get('FULL_MISS'), Delinquency, transform_features_dict)

ImputationTrainer = Delinquency.get('FULL_MISS_log')
IT = ImputationTrainer
to_impute =['MonthlyIncome','NumberOfDependents']
IT.impute = to_impute
# trainer, ModelTrains, models_to_run = ['RFR', 'DTR', 'KNNR', 'KNN', 'DT', 'RF','LR','ET','AB','GB'],
#                   cols=None, exclude_features=None, testsize=0.20, 
#                   results_dataframe=None, regress_only=False,
#                   filename='output'
XTrain = splitter(IT, Delinquency)




# # Keep a training set version which uses the binary versions for any variables that have missings
# train_derived_transformed_dropped = train_derived_transformed.copy()
# for feature in imputation_candidates:
#     del train_derived_transformed_dropped[feature]
#     try:
#         del train_derived_transformed_dropped[feature + '_log']
#     except:
#         print('ERROR:{}'.format(feature + '_log'))
# print('\nSummarize Training Set Version that Replaces Imputation Candidates with Binary Missings Variables:\n')
# train_derived_transformed_dropped
# train_derived_transformed_dropped.isnull().sum()


# ismiss_correlates_dict = get_correlates_dict(train_derived_transformed_dropped,
# train_derived_transformed_dropped.columns.tolist(),
# not_same=True, output_variable=['SeriousDlqin2yrs'])

# # Review variables that are most correlated in both versions of the transformed data.
# print("Review variable correlations for each version of the 'NON-MISSING' transformed training data:\n-Casewise drop of any examples with missings:")
# dropped_correlates_dict

# print("\n-Column wise drop of columns with missing examples, replaced by '_missing' binary variables:")
# ismiss_correlates_dict
# train_missing_transformed
# impute_correlates_dict = get_correlates_dict(train_missing_transformed, train_missing_transformed.columns.tolist(), not_same=True, output_variable='SeriousDlqin2yrs')
# train_missing_transformed.isnull().sum()
# impute_correlates_dict

# imputation_candidates

# train_missing_transformed.columns.tolist()

# # Build Model and Results Dictionaries:
# # Try predicting each imputation candidate using each version.
# training_version = {}
# print('\nCOMPARE CORRELATES OF KEY IMPUTATION CANDIDATES OF EACH VERSION OF TRAINING SET:\n')
# for feature in imputation_candidates:
#     print(dropped_correlates_dict[feature])
#     if feature in dropped_correlates_dict:
#         if 0 in training_version:
#             pass
#         else:
#             training_version[0] = {}
#         training_version[0][feature] = dropped_correlates_dict[feature]
#     if feature in ismiss_correlates_dict:
#         try:
#             if 1 in training_version:
#                 pass
#             else:
#                 training_version[1] = {}
#             print(ismiss_correlates_dict[feature])
#             training_version[1] = {feature : ismiss_correlates_dict[feature]}
#         except:
#             print('not in data')
#     if feature in impute_correlates_dict:
#         try:
#             if 2 in training_version:
#                 pass
#             else:
#                 training_version[2] = {}
#             print(impute_correlates_dict[feature])
#             training_version[2] = {feature : impute_correlates_dict[feature]}
#         except:
#             print('not in data')

# print(training_version)
    
# preprocessed_sets = ['train_transformed', 'train_derived_transformed_dropped']
# model_builder = {'trainees': preprocessed_sets,
#                  'test_sizes': [.20, .20],
#                  'C': {'RandomForestC' : {'n_estimators': ['NA', 10, 100], 'min_samples_split': ['NA', 1, 5, 10], 'class_weight' : ['NA', 'balanced', 'balanced_subsample']},
#                        'DecisionTreeC' : {'max_depth': ['NA', 100], 'min_samples_split':['NA', 1, 5, 10],
#                                          'class_weight' : ['NA', 'balanced']},
#                        'KNNC' : {'n_neighbors': [ 1, 2, 3, 5], 'weights': ['uniform', 'distance']}},
#                  'R': {'RandomForestR' : {'n_estimators':['NA', 10, 100],'min_samples_split': ['NA', 1, 5, 10]}, #,
#                        'DecisionTreeR' : {'max_depth': ['NA', 10, 100], 'min_samples_split':['NA', 1, 5, 10]},
#                        'KNNR' : {'n_neighbors': [ 1, 2, 3, 5], 'weights': ['uniform', 'distance']}},
#                  'versions':training_version
#                  }

# for aset in preprocessed_sets:
#     print(aset, "\n", eval(aset).shape)

# # Run models on ['MonthlyIncome','NumberOfDependents']
# to_impute =['MonthlyIncome','NumberOfDependents']
# to_avg = [c for c in train_transformed.columns.tolist() if c not in to_impute]
# outcome = 'SeriousDlqin2yrs'
# to_avg = [c for c in to_avg if c not in outcome]
# train_impute = train_missing_transformed.copy()


# # Select and apply classifier to the outcome variable on the dataset.
# # Now run imputed data through selected model
# features = ['RevolvingUtilizationOfUnsecuredLines_log',
#             'age', 'NumberOfTime30-59DaysPastDueNotWorse',
#             'DebtRatio','MonthlyIncome_log', 'NumberOfOpenCreditLinesAndLoans',
#             'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
#             'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

# # STEP 0
# ismissing_cols = ['MonthlyIncome_missing', 'NumberOfDependents_missing']
# fd0 = train_derived_transformed_dropped[ismissing_cols]
# just_impute_candy = train_missing_transformed.copy()
# fd0.isnull().sum()
# fd0.shape
# fd05 = pd.merge(fd0, just_impute_candy, left_index=True, right_index=True, how='left')
# fd05.isnull().sum()
# fd05.shape

# # STEP 1
# fd1 = fd05.fillna(fd05[to_avg].median())

# #Split data into people who reported the imputation feature versus those that didn't
# fd1[ismissing_cols].sum()

# for feature in to_impute:
#     check = feature + '_missing'
#     have_it = fd1[fd1[check]==0].copy()
#     have_it.drop(ismissing_cols, inplace=True, axis=1)
#     print(have_it.shape)
#     have_it.shape
#     dont_have_it = fd1[fd1[check]==1].copy()
#     dont_have_it.isnull().sum()
#     dont_have_it.drop(ismissing_cols, inplace=True, axis=1)
#     print(dont_have_it.shape)
#     # split
#     try:
#         os.mkdir(feature)
#     except:
#         print('DIRECTORY {} exists'.format(feature))
#     try:
#         os.mkdir(outcome)
#     except:
#         print('DIRECTORY {} exists'.format(outcome))
#     file = '{}/train_impute_'.format(feature) + feature 
#     results_df, best_model = splitter(feature, have_it, cols=impute_correlates_dict[feature], 
#                                       models_to_run = ['RFR', 'DTR', 'KNNR'], 
#                                       regress_only= True, filename=file)
#     # run R loop
#     # get best model
#     # predict clf to dont_have_it to replace missings there
#     # patch back together
#     # Impute with the best model for features with missings (Monthly Income and Number of Dependents).
#     imputer = clf 
#     imputer.fit(have_it[cols].as_matrix(), have_it[feature].as_matrix())
#     new_imputations = imputer.predict(dont_have_it[cols].as_matrix())
#     dont_have_it[feature] = new_imputations
    
#     combined = have_it.append(dont_have_it)
#     fd1[feature] = combined[feature]
#     # Updatelogs
#     checklog = feature + '_log'
#     if checklog in fd1.columns.tolist():
#         fd1[checklog] = fd1[feature].apply(lambda x: np.log(x) if x > 0 else np.log(x + 1))





# # finale = train_impute.copy()
# # finale.drop(drop_after, inplace=True, axis=1)
# # print(finale.columns)


# # Update model_builder for training on outcome.
# # model_builder['trainees'] = ['train_transformed']
# # outcome = 'SeriousDlqin2yrs'
# # file = 'train_impute_' + outcome
# # print('MODEL BUILDER DICTIONARY (before split_and_run)\n')

# # print_teacher(model_builder)

# outcome_results_df, outcome_best_model = split_and_run(model_builder, outcome,
#                                                        reg_or_clf= 'C', cols=features,
#                                                        filename=file, impute_dictionary=True,
#                                                        to_impute=to_impute)

# # # Compile selected classifier:
# # if outcome_best_model['Classifier']:
# #     clf = eval(outcome_best_model['Classifier'])()
# # if len(outcome_best_model['PARAMETERS'].keys()) > 0:
# #     for param in outcome_best_model['PARAMETERS']:
# #         clf.set_params(**{param:outcome_best_model['PARAMETERS'][param]})

# # classifier = clf
# # classifier.fit(train[features].as_matrix(), train[outcome].as_matrix())
# # probs = classifier.predict_proba(holdout[features].as_matrix())[::,1]
# # preds = classifier.predict(holdout[features].as_matrix())


# # print(classification_report(holdout[outcome], classifier.predict(holdout[features]), labels=[0, 1]))

# # # Compute confusion matrix
# # cm = confusion_matrix(np.array(holdout[outcome]),np.array(preds))
# # print(cm)
# # plot_confusion_matrix(cm)
# # plot_roc(outcome + '_roc_curve', holdout, probs, classifier, outcome)


# # # Other plots:

# # plt.close('all')
# # fig, ax = plt.subplots(figsize=(10,8))
# # delinquency_counts = pd.crosstab([train.age], train[outcome].astype(bool))
# # delinquency_counts.plot(kind = 'bar', stacked = True, color = ['blue','red'], grid = False, xticks = [])
# # xlabels = [20, 40, 60, 80, 100, 120]
# # xlabels = [a for a in range(0, 120) if a % 5 == 0]
# # plt.xticks(xlabels, xlabels)
# # plt.title('Delinquency Counts by Age')
# # plt.tight_layout()
# # fig.savefig('delinquency_counts_age')


# # plt.close('all')
# # fig, ax = plt.subplots(figsize=(10,8))
# # delinquency_counts = pd.crosstab([train.MonthlyIncome_log], train[outcome].astype(bool))
# # delinquency_counts.plot(kind = 'bar', stacked = True, color = ['blue','red'], grid = False, xticks = [])
# # xlabels = [20, 40, 60, 80, 100, 120]
# # xlabels = [a for a in range(0, 120) if a % 20 == 0]
# # plt.xticks(xlabels, xlabels)
# # plt.title('Delinquency Counts by MonthlyIncome_log')
# # plt.tight_layout()
# # fig.savefig('delinquency_counts_monthlyincome')
# # plot_roc(outcome + '_roc_curve', holdout, probs, classifier, outcome)