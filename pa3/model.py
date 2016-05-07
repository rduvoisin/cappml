import pandas as pd

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
    if to == 'pandas':
        if holdout_size:
            TRAIN, HOLDOUT = cross_validation.train_test_split(pd.read_csv(filename), test_size = holdout_size)
            labels = TRAIN.colums.tolist()
        else:
            labels, TRAIN = pd.read_csv(filename).columns, pd.read_csv(filename)
            if drop_column:
                del TRAIN[TRAIN.columns[drop_column]]
            return labels, TRAIN
    else:
        with open(filename) as f:
            labels = f.readline().strip().split(',')
            x = np.loadtxt(f, delimiter=',', dtype=np.float64)
            if holdout_size:
                TRAIN, HOLDOUT = cross_validation.train_test_split(data, test_size = holdout_size)
            else:
                TRAIN = x
                if drop_column:
                    del TRAIN[TRAIN.columns[drop_column]]
                return labels, TRAIN
    if drop_column:
        del TRAIN[TRAIN.columns[drop_column]]
    return labels, TRAIN, HOLDOUT


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
    
    def __init__(self, name, dataframe, outcome_name, testing_data = None):
        self._name = name
        # self._data, self.features = self._makeData(dataframe)
        self.__data_original = dataframe.copy()
        self._data = dataframe.copy()
        self._features = self._data.columns.tolist()
        self._outcome = outcome_name
        self._target = outcome_name
        self._toimpute = None # Made optional for broader application.
        self._parent = None
        self._validator = testing_data

    
    @property
    def name(self):
        '''
        Returns the Trainer's name id.'''
        return self._name

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
        if c not in self._data.columns.tolist():
            raise ValueError("{} is not a feature of Trainer object! \
                            So it can't be a target.".format(c))
        self._target = newtarget

    @property
    def parent(self):
        '''
        Returns the Trainer's parent Trainer 
        that was split to produce the Trainer.'''
        if self._parent:
            return self._parent

    @parent.setter
    def parent(self, newparent):
        '''Defines the parent Trainer'''
        if isinstance(newparent, Trainer):
            self._parent = newparent
        else:
            raise ValueError("Parent must also be a Trainer object!")

    @property
    def toimpute(self):
        '''
        Returns the list column names that require imputation.'''
        if self._toimpute:
            return self._toimpute

    @toimpute.setter
    def toimpute(self, column_list):
        '''Define the features that require imputation'''
        if isinstance(column_list, list):
            for c in column_list:
                if c not in self._data.columns.tolist():
                    raise ValueError("{} is not a feature of Trainer object!".format(c))
            self._toimpute = column_list
        else:
            raise ValueError("Imputation candidates must be a list of strings.")


    @property
    def outcome(self):
        '''
        Returns the feature that is the Trainer's 
        current modelling target.'''
        return self._outcome

    
    @property
    def now(self):
        '''Returns a the current Trainer's data.'''
        return self._data

    def set_data(self, newdataframe):
        '''Returns a the current Trainer's data.'''
        self._data = newdataframe.copy()


    def get_original_data(self,):
        '''Returns a the Trainer's data as it was first initialized.'''
        return self.__data_original
        

    def get_attributes(self):
        '''
        Returns the attributes of a single trainer object
        in the order required to store it as a ModelTrain
        in a ModelData object.
        '''
        return self._name, self._outcome, self._target, self._parent 


class ModelData(object):
    '''
    Stores the training and testing matrices of the data directory
    and stores the subsets of Trainer data with their respective
    best classifier.
    '''
    def __init__(self, index_of_dependent_variable, drop_column = False, holdout_size=0.10, title=None, task_number = None):
        ModelData.DEPENDENT_INDEX = index_of_dependent_variable
        ModelData.TITLE = title
        self.display = self.task(task_number)

        # Private attributes:
        self.__training_labels, \
        self.__training_data, \
        self.__holdout = read_file('data/training.csv', to='pandas', 
                                  drop_column = drop_column,
                                  holdout_size = holdout_size)
        self.__testing_labels, \
        self.__testing_data = read_file('data/testing.csv', to='pandas',
                                        drop_column = drop_column)
        self._training_y_values = self.__training_data[:, ModelData.DEPENDENT_INDEX] 
        self._testing_y_values = self.__testing_data[:, ModelData.DEPENDENT_INDEX]
        self._trainers, self._total_trainers = [], [] #self.__makeTrainers()
        self._model_names = None
        self._model_betas = None
        self._model_indices = None
        self._model_r2 = None

    
    @property
    def dependent_index(self):
        '''Returns the dependent variable's column number.'''
        return ModelData.DEPENDENT_INDEX

    
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

    
    # def __makeTrainers(self):
    #     '''
    #     Initializes model Trainer objects:
    #         * a name
    #         * a dataset
    #         * a target
    #         * possibly an analogous testing set
    #     Returns the list of initialized trainer objects.
    #     '''
    #     trainer_list = []
    #     # Can assume the label list to be indexed
    #     # with respect to their column number in the dataset.
    #     for column in range(len(self.__training_labels)):
    #         if column != ModelData.DEPENDENT_INDEX:
    #             k = trainer(self.__training_labels[column],
    #                           column,
    #                           self.__training_data[:, [column]],
    #                           self.__testing_data[:, [column]])
    #             trainer_list.append(k)
        # return trainer_list, len(trainer_list)
    

    def task(self, task):
        '''Output helper for printing task header.'''
        if task:
            self.display = "\n{} Task {}:".format(ModelData.TITLE, task)  
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


    def get_k(self, name_or_trainer_number = False):
        '''Returns a trainer object from either its name or its index number.'''
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