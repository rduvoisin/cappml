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
# inspect_pairplot(T, filedir=dataplotdir, inspect=inspect)

# Deal with missing values.
# Collect candidates for imputations of missings. 
feature_wnull = list_features_wmissing(T.now.copy())
print('\n', T.name, 'Correlates\n', T.now.columns)
correlated_features = get_correlates_dict(T, feature_wnull, output_variable=T.outcome)
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

print('\nSummarized Data After Tagging Cases with Missing Values:\n')
# inspect_correlations(Delinquency)

# Save tracer datasets as Trainer objects.(name, dataframe, outcome_name, validator = None, ModelTrainIndex = None))
transform_features_dict = {'RevolvingUtilizationOfUnsecuredLines' : 'log',
                           'MonthlyIncome' : 'log', 'DebtRatio': 'log'}
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

# Recall train_derived_transformed (contains missings binaries)
# train_derived_transformed.isnull().sum()
for t in last2trainers:
    print(t.id, t.name, t.shape, t.parent.name, t)


# Decide which dataset to use for imputation:
# Corrected TRAIN plus any Transformed features:
gen_transform_data(Delinquency.get('FULL_MISS'), Delinquency, transform_features_dict)

ImputationTrainer = Delinquency.get('FULL_MISS_log')
ITdata = ImputationTrainer.now
for k in transform_features_dict:
    ITdata.drop(k, inplace=True, axis=1)
ImputationTrainer.set_data(ITdata)

IT = ImputationTrainer
# to_impute =['MonthlyIncome','NumberOfDependents']
# IT.impute = to_impute

best['Estimator'], \
best['PARAMETERS'], \
cols, \
imputation_stats_and_methods, best = splitter(IT, Delinquency, models_to_run=['RF', 'KNN', 'DT'])
print('BEST\n', best)
print('imputation_stats_and_methods\n', imputation_stats_and_methods)