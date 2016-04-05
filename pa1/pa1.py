# Rebeccah Duvoisin

# Homework 1

# 	Due: 5pm April 4, 2016

# 	You are being provided with data in a CSV file that contains student records. Each row is a record about a student containing their name, demographic information, grades, test score, attendance, and whether they graduated or not.

# Problem A

#     The first task is to load the file and generate summary statistics 
#     for each field as well as probability distributions or histograms. 

#     You will notice that a lot of students are missing gender values . 
#     Your task is to infer the gender of the student based on their name. 
#     Please use the API at www.genderize.io to infer the gender of each student 
#     and generate a new data file.

#     You will also notice that some of the other attributes are missing. 
#     Your task is to fill in the missing values for 
#     Age, GPA, and Days_missed 
#     using the following approaches:

#     Fill in missing values with the 
#     mean of the values for that attribute
    
#     Fill in missing values with a class-conditional mean 
#     (where the class is whether they graduated or not).
#     Is there a better, more appropriate method for 
#     filling in the missing values? 
#     If yes, describe and implement it. 

# 	You should create 2 new files with the missing values filled, 
# 	one for each approach A, B, and C and submit those along with your code. 

	# Please submit the Python code for each of these tasks as well as 
	# the new data files for this assignment.
# from __future__ import division
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

base_url = "https://api.genderize.io"
index_ref  = "/?name="

def grab_link_like_person(url, interval=2):
    '''
    Clicks a url address and sets of timer
    to ensure a specified rate limit for url requests
    to the target web server.

    Returns an opened url page to scrape  content.
    '''
    response = requests.get(url)
    time.sleep(interval)
    print(response)
    return response


def grab_gender(row, interval=2, column='First_name', reference_column='Gender', guess_column='Guess'):
    '''
    Requests genderize.io for each row where
    gender is missing (np.nan). 
    Converts answer into a response dictionary
    and returns the 'gender' value to the cell.
    '''
    if isinstance(row[reference_column], float):
        requesturl = base_url + index_ref + row[column]
        responses = json.loads(grab_link_like_person(requesturl, interval).text)
        print(responses['name'], row[guess_column], responses['gender'].title())
        row[guess_column] = responses['gender'].title()
        print(responses['name'], row[guess_column])
    else:
        row[guess_column] = row[guess_column]


def convert_na_by_class_mean(row, column, unique_classes=None,
                             student_data=None, feature_by_class_mean=None,
                             reference_column='Grad', INDEX='ID'):
    '''
    Converts missing values with respect to class means.
    '''
    if not unique_classes:
        unique_classes = student_data[reference_column].unique().tolist()
    feature_by_class_mean = student_data.groupby(reference_column)[column].agg('mean').copy()
    if str(row[column])=="nan":
        for level in unique_classes:
            f_mean = feature_by_class_mean[level]
            if row[reference_column] == level:
                # print('Mean for level ({}={}) for {}: {}\n'.format(reference_column, level, column, f_mean))
                row[column] = f_mean
                return row[column]
    else:
        return row[column]


if __name__=='__main__':
    # Read in csv data
    student_csv = 'mock_student_data.csv'
    student_data = pd.read_csv(student_csv)
    student_data['Male']=student_data.Gender.map({'Female' : 0, 'Male' : 1})
    student_data['Grad']=student_data.Graduated.map({'No' : 0, 'Yes' : 1})
    student_data.head(10)
    student_data.shape  
    # Show some summary statistics:
    #     The summary statistics should include 
    #     mean, 
    #     median, 
    #     mode, 
    #     standard deviation, as well as 
    #     the number of missing values for each field.
    student_data.describe()
    for feature in student_data.columns.tolist():
    	print('\nInspect {}:'.format(feature), student_data[feature].describe())

    # Retrieve Genderized Gender for all missing Gender values:
    #     You will notice that a lot of students are missing gender values . 
    #     Your task is to infer the gender of the student based on their name. 
    #     Please use the API at www.genderize.io to infer the gender of each student 
    #     and generate a new data file.
    # 	GET https://api.genderize.io/?name[0]=peter&name[1]=lois&name[2]=stevie 

    # first = 'First_name'
    # row = 'Rebeccah'
    # requesturl = base_url + index_ref + row
    # response = grab_link_like_person(requesturl) //still timed out...

    # student_data['Guess'] = student_data['Gender']
    # student_data2 = student_data.copy()
    # student_data3 = student_data.copy()
    # student_data4 = student_data.copy()
    # student_data4['Guess'] = student_data2.apply(lambda row: grab_gender(row,2), axis=1)

    # student_data2.head(10)
    # student_data4.head(10)

#     You will also notice that some of the other attributes are missing. 
#     Your task is to fill in the missing values for 
#     Age, GPA, and Days_missed 
#     using the following approaches:

#     Fill in missing values with the 
#     mean of the values for that attribute

    student_data_na_byfeaturemean = student_data.copy()
    with_null_data = ['Age', 'GPA', 'Days_missed']
    for feature in with_null_data:
        print(student_data_na_byfeaturemean[feature].describe())
        f_mean = student_data_na_byfeaturemean[feature].mean()
        print('Mean for {}: {}\n'.format(feature, f_mean))
        student_data_na_byfeaturemean[feature] = student_data_na_byfeaturemean[feature].replace(np.nan, f_mean)
        print(student_data_na_byfeaturemean[feature].describe())

#     Fill in missing values with a class-conditional mean 
#     (where the class is whether they graduated or not).
#     Is there a better, more appropriate method for 
#     filling in the missing values? 
#     If yes, describe and implement it.

    print('\n\nReplace missings by class means:\n')
    reference_column = 'Grad'
    student_data_byclass_mean = student_data.copy() 
    for feature in with_null_data:
        print('\n\n{}:\n'.format(feature))
        unique_classes = student_data[reference_column].unique().tolist()
        feature_by_class_mean = student_data.groupby(reference_column)[feature].agg('mean').copy()
        student_data_byclass_mean[feature] = \
            student_data_byclass_mean.apply(lambda row: 
            convert_na_by_class_mean(row, feature, 
            unique_classes=unique_classes, 
            feature_by_class_mean=feature_by_class_mean,
            student_data=student_data), axis=1)
        print(student_data_byclass_mean[feature].describe())


    print("We should instead try to correlate whether the \
          presence of data on a feature is correlated with the outcome. \
          We may also choose to stratify further. Then we may \
          want to use sample weights and means. One way to do this \
          is propensity score matching.")


    # Problem B
    #
    # A larger data set than the one in the previous problem 
    # was used to build a logistic regression model that predicts 
    # the probability an individual student will graduate.
    #
    # Below are coefficients from this model.  
    # The definitions of the variables are below.
    #
    #     Male-Coded as 1 if the student is M, 0 if he/she is not
    #     Female-Coded as 1 if the student is Female, 0 if he/she is not
    #     Ln_Family_Income-The natural logarithm of the student's Family income (in dollars)
    #     Age-The student's age (in years)
    #     Age_Sq-The student 's age (in years) squared
    #     Census_College-The percentage of residents in the student 's neighborhood who have a college degree (scaled from 0 to 100)
    #     AfAm-Coded as 1 if the student is African American, 0 if he/she is not
    #     AfAm_Male-Coded as 1 if the student is both African American and Male, 0 if he/she is not
    #     Constant-The constant term
    #
    #
    #
    #     Consider 4 students, Adam, Bob, Chris and David. 
    # 		Adam and Chris share identical characteristics except for 
    # 		their family incomes.  Bob and David also share identical 
    # 		characteristics (with each other, not necessarily Adam and Chris), 
    # 		except for their incomes.
    #

    #
    # Based on the coefficients above, who would you think has a higher probability of graduating?
    #
    #     Chris
    #     David
    #     They have the same probability
    #     Cannot tell based on the information provided
    #
    # What is your reasoning?  (you need not calculate an exact probability to answer this question. Just explain your reasoning in general terms.)
    #
    #     The coefficient for AfAm_Male is negative. How do you interpret this? Does this mean that African-American Males are more likely to not graduate than African-American Females? What about relative to non African American males?
    #     How do we interpret the difference in graduation probability between students of different ages? How do the variables in the model estimate such probability?
    #     Are there any variables in this model that you would choose to drop? Why or why not? Would you need more information in order to make this decision?
    #
    # Instruction of how to submit your homework is found at http://people.cs.uchicago.edu/~larsson/capp30254-spr-15/. The instructions assume familiarity with the submission system used in CMSC 12100 and 12200. If you did not take these classes, please come to office hours for help getting your assignment submitted.
    # mock_student_data.csv
