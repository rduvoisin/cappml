from __future__ import division 
<<<<<<< HEAD
from model import *
# from mlpipeline_pa3 import list_features_wmissing
from sklearn import cross_validation
=======
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
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
import notebook
import re
import seaborn as sns; sns.set(style="ticks", color_codes=True)
 
# %matplotlib inline
FIGWIDTH = 10
FIGHEIGHT = 8

<<<<<<< HEAD


=======
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
def inspect_correlations(ModelTrains, filedir='data/plots'):
    '''Produce Correlation Matrices with Nonmissing Traner Objects'''
    plt.close('all')
    nonissing_trainers = []
<<<<<<< HEAD
    print('Nonmissing Trainers for Correlations')
    for trainer in ModelTrains.trainers:
        save_this_directory = filedir + '/{}'.format(trainer.name)
        save_this_here = save_this_directory + '/correlations'
        try:
            os.mkdir(filedir)
        except:
            pass
        try:
            os.mkdir(save_this_directory)
        except:
            pass
        try:
            os.mkdir(save_this_here)
        except:
            pass
        if trainer.now.isnull().sum().sum() == 0:
            x, y, z, a = trainer.get_attributes()
            nonissing_trainers.append(trainer)
            # fig, axs = plt.subplots(figsize=(12, FIGWIDTH))
            plt.figure(figsize=(12, FIGWIDTH))
            g = sns.corrplot(trainer.now, annot=False)
            t = "Non-Missing Correlation Matrix of {}, {}".format(trainer.name, trainer.shape)
            doc = '{}/{}.png'.format(save_this_directory,'corrplot_{}'.format(trainer.shape))
            # g.fig.text(0.33, 1.02, t, fontsize=18)
            # g.savefig(doc)
=======
    save_this_directory = filedir + '/{}'.format(t.name)
    save_this_here = save_this_directory + '/correlations'
    try:
        os.mkdir(filedir)
    except:
        pass
    try:
        os.mkdir(save_this_directory)
    except:
        pass
    try:
        os.mkdir(save_this_here)
    except:
        pass
    print('Nonmissing Trainers for Correlations')
    for trainer in ModelTrains.trainers:
        if trainer.now.isnull().sum().sum() == 0:
            x, y, z, a = t.get_attributes()
            nonissing_trainers.append(trainer)
            plt.figure(figsize=(12, FIGWIDTH))
            g = sns.corrplot(trainer.now, annot=False)
            t = "Non-Missing Correlation Matrix of {}, {}".format(trainer.name, t.shape)
            doc = '{}/{}.png'.format(save_this_directory,'corrplot_{}'.format(t.shape))
            g.fig.text(0.33, 1.02, t, fontsize=18)
            g.savefig(doc)
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
            plt.close('all')
    plt.close('all')



def inspect_pairplot(trainer, descriptor=None, filedir='data/plots', 
                    x_vars=None, y_vars=None, 
                    inspect=None, size=None):
    '''Produce pairwise histograms'''
    plt.close('all')
    if not inspect:
        inspect = trainer.now.columns.tolist()
    if trainer.target not in inspect:
        inspect.append(trainer.target) 
    x = len(inspect)
    if not size:
        size = x 
        aspect = 1
    g = sns.pairplot(data=trainer.now[inspect].dropna(), x_vars=x_vars, y_vars=y_vars, 
        hue = trainer.target, palette="Set1", 
        aspect= aspect, size=size)
    save_this_directory = filedir + '/{}'.format(trainer.name)
    save_this_here = save_this_directory
    try:
        os.mkdir(filedir)
    except:
        pass
    try:
        os.mkdir(save_this_directory)
    except:
        pass
    try:
        os.mkdir(save_this_here)
    except:
        pass
    if descriptor:
        desc = descriptor.replacer
    t = "Distributions of {}/{} features of {} by {}".format(x, 
                                                            len(trainer.now.columns.tolist()), 
                                                            trainer.name,
                                                            trainer.target)
    doc = '{}/{}.png'.format(save_this_here,'pair_plot_{}'.format(x))
    g.fig.text(0.33, 1.02, t, fontsize=18)
    g.savefig(doc) 
    plt.close('all')


def inspect_zeros(trainer, filedir, inspect=None, FIGWIDTH=FIGWIDTH, FIGHEIGHT=FIGHEIGHT):
    '''Produce side-by-side log histograms.'''
    plt.close()
    complete = []
    D = trainer.now.copy()
    if not inspect:
        inspect = D.columns.tolist()
    save_this_directory = filedir + '/{}'.format(trainer.name)
    save_this_here = save_this_directory + '/zeros'
    try:
        os.mkdir(filedir)
    except:
        pass
    try:
        os.mkdir(save_this_directory)
    except:
        pass
    try:
        os.mkdir(save_this_here)
    except:
        pass
    for feature in inspect:
        print('Inspect {} for Zeros'.format(feature))
        plt.close()
        for x in inspect:
            if x != feature and (x, feature) not in complete:
                compare = (x, feature)
                complete += [compare]
                try:
                    fig, axs = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
                    np.log1p(D[D[feature] == 0][x]).hist(bins=30, label ='{} == 0'.format(feature), normed=True)
                    np.log1p(D[D[feature] > 0][x]).hist(bins=30, label ='{} > 0'.format(feature), normed=True).legend(loc='upper right')
                    t = "Log {} | {} = Zero.".format(x, feature)
                    plt.title(t)
                    axs.grid(False)
                    doc = '{}/{}.png'.format(save_this_here,'inspect_{}_when_{}_zero'.format(x, feature))
                    plt.gcf().tight_layout()
                    plt.savefig(doc) 
                except:
                    plt.close('all')
                    pass
                plt.close()
        
        fig, axs = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT)) 
        tag = '{}_pairplot_when_zero'.format(feature)
        t = "Distribution | {} = Zero.".format(feature)
        doc = '{}/{}.png'.format(save_this_here, tag)  
        try:
            g = sns.pairplot(data=D[D[feature]==0][inspect].dropna(), 
                             hue = trainer.target, palette="Set1")
            g.fig.text(0.33, 1.02, t, fontsize=20)
            g.savefig(doc) 
        except:
            plt.close('all')
            pass
        plt.close('all')


def read_data(filename, to='pandas', holdout_size = False, drop_column = False):
    '''
    Read data from the specified file.  Split the lines and convert
    float strings into floats.  Assumes the first row contains labels
    for the columns.

    Inputs:
      filename: name of the file to be read

    Returns:
      (list of strings, pandas dataframe)
    '''
    # DF.drop([DF.columns[[0, 1, 3]]], axis=1) # Note: zero indexed
    # DF.drop('column_name', axis=1, inplace=True)
    if to == 'pandas':
        data = pd.read_csv(filename)
        if isinstance(drop_column, int):
            data = data.drop([data.columns[drop_column]], axis=1)
        elif isinstance(drop_column, string):
            if drop_column in data.columns:
                data = data.drop(drop_column, axis=1, inplace=True)
        if holdout_size:
            TRAIN, HOLDOUT = cross_validation.train_test_split(data, test_size = holdout_size)
            labels = TRAIN.columns.tolist()
        else:
            labels, TRAIN = data.columns, data.copy()
            HOLDOUT = pd.DataFrame()
    else:
        with open(filename) as f:
            labels = f.readline().strip().split(',')
            data = np.loadtxt(f, delimiter=',', dtype=np.float64)
            if holdout_size:
                TRAIN, HOLDOUT = cross_validation.train_test_split(data, test_size = holdout_size)
            else:
                TRAIN = data.copy()
                HOLDOUT = pd.DataFrame()
                # if drop_column:
                #     del TRAIN[TRAIN.columns[drop_column]]
    return labels, TRAIN, data, HOLDOUT


class Trainer(object):
    '''
    Initializes a dataset as a trainer object, which contains:
        * a name
        * a pandas DataFrame = data
        * an outcome name (the dependent feature)
        * a target for modelling (outcome_name by default)
        * optionally a heritage parent
        * optionally a validation DataFrame
    '''
    
    def __init__(self, name, dataframe, outcome_name, validator = None, ModelTrainIndex = None, parent = None):
        self._name = name
        # self._data, self.features = self._makeData(dataframe)
        self.__data_original = self.__makeData(dataframe)
        self.__last_version = self.__makeData(dataframe)
        self._data = self.__makeData(dataframe)
        self._features = self._data.columns.tolist()
        self._outcome = outcome_name
        self._target = outcome_name
        self._changes = 0
<<<<<<< HEAD
        self._median = {}
        self._toimpute = [] 
        self._missings = self.__updateMissings()
        self._transformed = {}
        self._transformations = []
=======
        self._toimpute = None # Made optional for broader application.
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
        self._parent = parent
        self._number = ModelTrainIndex
        self._children = []
        self._validator = validator
        self._imputed = []
        self._best_estimators = {}

    
    @property
    def name(self):
        '''Returns the Trainer's name id.'''
        return self._name


    @property
    def best(self):
        '''Get this Trainer's dictionary of 
        best classifiers obtained for any modelling target
        that has been obtained from these Trainer data.'''
        return self._best_estimators
    

    @best.setter
    def best(self, pair):
        '''Set a key and estimator model in the best models dict'''
        if isinstance(pair, (tuple, list)):
            self._best_estimators[pair[0]] = pair[1]
        elif isinstance(pair, dict):
            key = pair.keys().tolist()[0]
            self._best_estimators[key] = pair[key]
        else:
            raise ValueError('Must pass a binary tuple, list, or dictionary')


    @property
    def id(self):
        '''Returns the Trainer's ordinal number within a ModelTrain's trainer_list.'''
        return self._number


    @id.setter
    def id(self, number):
        '''Modifies the Trainer's ordinal number to it's index a ModelTrain's trainer_list.'''
        self._number = number

    @property
    def outcome(self):
        '''
        Returns the feature that is the Trainer's 
        current modelling target.'''
        return self._outcome

    @property
    def target(self):
        '''
        Returns the feature that is the Trainer's 
        current modelling target.'''
        return self._target

    @target.setter
    def target(self, newtarget):
        '''
        Sets the feature that is the Trainer's 
        current modelling target.'''
        if newtarget not in self._data.columns.tolist():
            raise ValueError("{} is not a feature of Trainer object! \
                            So it can't be a target.".format(newtarget))
        self._target = newtarget

    @property
    def validator(self):
        '''Returns the Trainer's current hold out set.'''
        return self._validator


    @validator.setter
    def validator(self, newtrainer):
        '''Set's the Trainer's hold out set.'''
        if isinstance(newtrainer, Trainer):
            self._validator = newtrainer


    @property
    def parent(self):
        '''
        Returns the Trainer's parent Trainer 
        that was split to produce the Trainer.'''
        if self._parent:
            return self._parent

    def set_parent(self, newparent, ModelTrains):
        '''Defines the parent Trainer and adds as Child of parent'''
        if isinstance(newparent, Trainer):
            self._parent = newparent
            ModelTrains.get(newparent.name).add_child(ModelTrains.get(self._name))
        else:
            raise ValueError("Parent must also be a Trainer object!")
    
    @property
    def shape(self):
        '''Returns the current shape of Trainer object's data.'''
        return self._data.shape

   
    def nulls(self):
        '''Returns a series of Trainer object's sums of null data.'''
        return self._data.isnull().sum()


    @property
    def impute(self):
        '''Returns the list column names that require special imputation.'''
        return self._toimpute

    @impute.setter
    def impute(self, column_list):
        '''Define the features that require imputation modelling'''
        if isinstance(column_list, list):
            for c in column_list:
                if c not in self._data.columns.tolist():
                    raise ValueError("{} is not a feature of Trainer object!".format(c))
                elif self._data[c].isnull().sum().sum() == 0:
                    raise ValueError("{} has no missing to impute in {}!".format(c, self._name))
                else:
                    self._toimpute = column_list
        else:
            raise ValueError("Imputation candidates must be a list of column strings.")
        self.__updateMissings()


    @property
    def imputed(self):
        '''Returns the list column names that have been fit and imputed.'''
        return self._imputed

    @imputed.setter
    def imputed(self, name):
        '''Add feature to a list of imputed features.'''
        if name not in self._data.columns.tolist():
            raise ValueError("{} is not a feature of Trainer object!".format(name))
        elif self._data[c].isnull().sum().sum() > 0:
            raise ValueError("{} has HAS MISSING DATA STILL to impute in {}!".format(name, self._name))
        else:
            self._imputed.append(name) 
        self._missings = self.__updateMissings()

    
    @property
    def missing(self):
        '''Return a list columns with missing data that are 
        not already staged for imputation modelling.'''
        self._missings = self.__updateMissings()
        return self._missings


    def __updateMissings(self):
        '''Updates the record of columns with missing values.'''
        has_missings = self.list_features_wmissing(self._data)
        missings_list = []
        for c in has_missings:
            if c not in self.impute:
                missings_list.append(c)
        return missings_list


    @property
    def transformed(self):
        '''
        Returns the dictionary of transformed 
        features to their respective transformation features.'''
        if self._transformed:
            return self._transformed

    def transform(self, pair):
        '''
        Maps the features to the features that are its transformations 
        in order to identify invalid intercorrelations.'''
        if isinstance(pair, dict):
            for k in pair.keys():
                if k in self._data.columns:
                    if k not in self._transformed.keys():
                        self._transformed[k] = pair[k]
                    else:
                        print("Adding {} to list of {} transformations {}.".format(pair[k], k, self._transformed[k]))
                        if isinstance(self._transformed[k], str):
                            self._transformed[k] = [self._transformed[k]]
                        self._transformed[k].append(pair[k])
                else:
                    raise KeyError("{} is not a feature of {}.".format(k, self._name))


    @property
    def children(self):
        '''Returns a list of Trainer's children Trainer's.'''
        return self._children


    def add_child(self, newchild):
        '''
        Adds a Child Trainer Object to the Children list.
        '''
        if isinstance(newchild, Trainer):
            # print('\n**{} birthed {}!'.format(self.name, newchild.name))
            self._children.append(newchild)


    def child(self, names):
        '''Returns a list of Child Trainer Objects specified by name strings.'''
        child_list = []
        if isinstance(names, string):
            for child in self._children:
                    if name == child.name:
                        child_list.append(child)
        if isinstance(names, list):
            for name in names:
                for child in self._children:
                    if name == child.name:
                        child_list.append(child)
        return child_list

    
    def __makeData(self, dataframe):
        '''Constuctor for initializing Trainer data.'''
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Data must be passed in a dataframe!")
        else:
            return dataframe.copy()



    
    @property
    def now(self):
        '''Returns a the current Trainer's data.'''
        return self._data
  

    @property
    def same(self):
        '''Returns the number of times the Trainer's data have been rewritten.'''
        return self._data.equals(self.__last_version)


    @property
    def changes(self):
        '''Returns the number of times the Trainer's data have been rewritten.'''
        return self._changes


    def set_data(self, newdataframe):
        '''Returns a the current Trainer's data.'''
        if isinstance(newdataframe, pd.DataFrame):
            if not self._data.equals(newdataframe):
                self.__last_version = self._data.copy()
                self._changes += 1
                self._data= newdataframe


    def reset(self, swap=False):
        '''Resets a the current Trainer's data to the last version.'''
        if swap:
            fast_forward = self._data.copy()
            self._data = self.__last_version.copy()
            self.__last_version = fast_forward.copy()
        else:
            self._data = self.__last_version.copy()
        self._changes += 1


    def get_original_data(self,):
        '''Returns a the Trainer's data as it was first initialized.'''
        return self.__data_original
    

    def get_attributes(self):
        '''
        Returns the attributes of a single trainer object
        in the order required to store it as a ModelTrain
        in a ModelTrains object.
        '''
        children_names = []
        for child in self._children:
            children_names = children_names.append(child.name)
        parental_object = None
        parental_name = None
        parental_shape = None
        if self._parent:
            parental_object = self._parent
            parental_name = parental_object.name
            parental_shape = parental_object.shape

        print('\n\tTrainer Name: {}, Shape = {}, ModelTrainIndex Position: {}\n\
              \tParent Name: {}, Shape = {}\n\
              \tChildren = {}\n\
              \tOutcome = {}\n\
              \tTarget = {}\n\
              \tSum of Nulls: \n{}\n\
              \tReturning >> Name, Outcome, Target, Parent\n'.\
              format(self._name, self.shape, self.id, parental_name, parental_shape, children_names,
                    self._outcome, self._target, self.nulls()))
        return self._name, self._outcome, self._target, parental_object

    @staticmethod
    def list_features_wmissing(dataset):
        '''
        Return all features that have missing values:
            - a list of just those features.'''
        print('Summary Statistics on Full Data set:\n{}'.format(dataset.describe(include='all').round(2)))
        has_null = pd.DataFrame({'Total_missings' : dataset.isnull().sum()})
        has_null[(has_null.Total_missings >0)].index.tolist()
        print('\n\n{} Features containing missing values: {}\n'
              .format(len(has_null[(has_null.Total_missings >0)].index.tolist()),
               has_null[(has_null.Total_missings >0)].index.tolist()))
        print(has_null.ix[2:,:])
        return has_null[(has_null.Total_missings >0)].index.tolist()


class ModelTrains(object):
    '''
    Stores the training and testing matrices of the data directory
    and stores the subsets of Trainer data with their respective
    best classifier.
    '''
    def __init__(self, filedir, index_of_dependent_variable=0, drop_column = False, holdout_size=0.10, title=None, task_number = None):
        ModelTrains.DEPENDENT_INDEX = index_of_dependent_variable
        ModelTrains.TITLE = title
        self.display = self.task(task_number)

        # Private attributes:
        self.__training_labels, \
        self.__training_data, \
        self.__full_training_data, \
        self.__holdout = read_data('{}/training.csv'.format(filedir), to='pandas', 
                                  drop_column = drop_column,
                                  holdout_size = holdout_size)
        self.__testing_labels, \
        self.__testing_data, \
        self.__full_testing_data, \
        self.__testing_holdout = read_data('{}/testing.csv'.format(filedir), to='pandas',
                                              drop_column = drop_column)
        self._training_y_values = self.__training_data[self.__training_data.columns[ModelTrains.DEPENDENT_INDEX]]
        self._testing_y_values = self.__testing_data[self.__testing_data.columns[ModelTrains.DEPENDENT_INDEX]]
        self._trainers, self._total_trainers = self.__makeTrainers()
        self._model_names = None
        self._model_betas = None
        self._model_indices = None
        self._model_r2 = None

    
    @property
    def dependent_index(self):
        '''Returns the dependent variable's column number.'''
        return ModelTrains.DEPENDENT_INDEX

    
    @property
    def training_y_values(self):
        '''Returns dependent array of the training_data.'''
        return self._training_y_values 


    @property
    def testing_y_values(self):
        '''Returns dependent array of the testing_data.'''
        return self._testing_y_values


    @property
    def training_data(self):
        '''Returns the arrays of the training data matrix.'''
        return self.__training_data


    @property
    def testing_data(self):
        '''Returns the arrays of the testing data matrix.'''
        return self.__testing_data


    @property
    def trainers(self):
        '''Lists all the trainer objects of the data.'''
        if isinstance(self._trainers, list):
            # Return a copy so that this list can't be 
            # manipulated via property.
            return self._trainers[:]
        else:
            return [self._trainers]


    @property
    def total_trainers(self):
        '''Returns the total number of trainers.'''
        return self._total_trainers


    @property
    def model_names(self):
        '''A string of names of the trainers in the current model in memory.'''
        return self._model_names

    
    @property
    def model_indices(self):
        '''Lists the column numbers of the trainers in the current model.'''
        if isinstance(self._model_indices, list):
            # Return a copy so that this list can't be 
            # manipulated via property.
            return self._model_indices[:]
        else:
            return [self._model_indices]


    @property
    def model_r2(self):
        '''Returns the R-squared value of the current model in memory.'''
        return self._model_r2


    @property
    def model_betas(self):
        '''Returns the estimated coefficients of the current model in memory.'''
        return self._model_betas

    
    def __makeTrainers(self):
        '''
        Initializes Read in Data as Trainer objects:
            * a name
            * a dataset
            * a target
            * possibly an analogous testing set
        Returns the list of initialized trainer objects.
        Trainer(name, dataframe, outcome_name, validator = None, ModelTrainIndex = None)
        '''
        trainer_list = []
        # Make Trainer objects from the datasets and store each one.
        if not self.__full_training_data.empty:
            outcome_name= self.__training_labels[ModelTrains.DEPENDENT_INDEX]
            TRAIN_PARENT = Trainer('TRAIN_PARENT', self.__full_training_data.copy(), outcome_name, ModelTrainIndex=len(trainer_list))
            pname, poutcome, ptarget, pparent = TRAIN_PARENT.get_attributes()
            trainer_list.append(TRAIN_PARENT)
            if not self.__training_data.empty:
                if not self.__holdout.empty:
                    HOLDOUT = Trainer('HOLDOUT', self.__holdout.copy(), outcome_name, 
                                     ModelTrainIndex=len(trainer_list), parent=TRAIN_PARENT)
                    hname, houtcome, htarget, hparent = HOLDOUT.get_attributes()
                    trainer_list.append(HOLDOUT)
                    TRAIN = Trainer('TRAIN', self.__training_data.copy(), outcome_name, 
                                    validator=HOLDOUT, ModelTrainIndex=len(trainer_list), 
                                    parent=TRAIN_PARENT)
                else:
                    TRAIN = Trainer('TRAIN', self.__training_data.copy(), outcome_name, 
                                    ModelTrainIndex=len(trainer_list), parent=TRAIN_PARENT)
                TRAIN_PARENT.add_child(TRAIN)
                trname, troutcome, trtarget, trparent = TRAIN.get_attributes()
                trainer_list.append(TRAIN)
            if not self.__testing_data.empty:
                TEST = Trainer('TEST', self.__testing_data.copy(), outcome_name, 
                               ModelTrainIndex=len(trainer_list), parent=TRAIN_PARENT)
                TRAIN_PARENT.add_child(TEST)
                tename, teoutcome, tetarget, teparent = TEST.get_attributes()
                trainer_list.append(TEST)
        return trainer_list, len(trainer_list)
    

    def task(self, task):
        '''Output helper for printing task header.'''
        if task:
            self.display = "\n{} Task {}:".format(ModelTrains.TITLE, task)  
        else:
            self.display = "\n"
        return self.display


    def set_model(self, beta_estimates, r2, columns_numbers):
        '''
        Validates and stores the results of a model 
        as the current model in memory.
        '''
        if isinstance(beta_estimates, np.ndarray):
            self._model_betas = beta_estimates
        else:
            raise ValueError("Coefficients must be numpy arrays.")
        if isinstance(r2, (int, float)):
            self._model_r2 = r2
        else:
            raise ValueError("R2 must be numeric.")
        if isinstance(columns_numbers, (int, list)):
            name_list = []
            if isinstance(columns_numbers, list):
                model_indices = columns_numbers
                for num in columns_numbers:
                    for k in self.trainers:
                        if k.column == num:
                            name_list.append(k.name)
                            break
            if isinstance(columns_numbers, int):
                model_indices = [columns_numbers]
                for k in self.trainers:
                    if k.column == columns_numbers:
                        name_list.append(k.name)
                        break
            sep = ", "
            self._model_names = sep.join(name_list)
            self._model_indices = model_indices
        else:
            raise ValueError("Column numbers must be int or list of integers.")


    def get(self, name_or_trainer_number = False):
        '''Returns a trainer object from either its name or its trainer_list index number.'''
        if name_or_trainer_number:
            if isinstance(name_or_trainer_number, str):
                for k in self.trainers:
                    if k.name == name_or_trainer_number:
                        return k
            elif isinstance(name_or_trainer_number, int):
                count = 0
                for k in self.trainers:
                    count += 1
                    if count == name_or_trainer_number:
                        return self.trainers[count]
            else:
                raise ValueError("Provide a valid trainer name or list number.")

    def add(self, newtrainer):
        '''
        Adds a new Trainer object to the staging area only if data is different:
            * a name
            * a dataset
            * a target
            * possibly an analogous testing set
        Returns the list of initialized trainer objects.
        '''
        trainer_list = self.trainers
        if isinstance(newtrainer, Trainer):
            for t in trainer_list:
                if t.now.equals(newtrainer.now):
                    raise ValueError("\n{} Redundant! {} already contains this data.".format(newtrainer.name, t.name))
                    return
            newtrainer.id = len(trainer_list)
            self._trainers.append(newtrainer)

<<<<<<< HEAD
    def show(self):
        '''
        Print Trainers
        '''
        print('\nCURRENT TRAINER OBJECTS')
        for i in self._trainers:
            print(i.name, i.shape, '#{}'.format(i.id))

=======
>>>>>>> b03c6e05dcda2f6fd6961d9b2a3185f2dec10593
