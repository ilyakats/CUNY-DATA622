# -*- coding: utf-8 -*-
"""
CUNY MSDS Program, DATA 622, Homework 2
Created: September 2018

@author: Ilya Kats

This module reads Titanic data set from public GitHub.
"""

import pandas as pd

# Define data location here for ease of modification
root = 'https://raw.githubusercontent.com/ilyakats/CUNY-DATA622/master/HW2/TitanicData/'

def read_data_set(set_name):
    """Read data from GitHub and return data frame."""
    
    # Confirm that set name is something expected
    if set_name not in ['train','test']:
        raise ValueError('Set name must be either train or test')
    
    # Read the CSV file
    try:
        df = pd.read_csv(root+set_name+'.csv')
    except:
        print('Error downloading the data set')
        raise

    # Return data frame if successfully reached this point
    return df

def validate_data_set(df, target_exists=True):
    """Validate that data frame is in expected format and 
    some data exist.
    
    Expectations:
        - At least one row
        - 12 columns (11 if test set)
        - Columns are named as expected
    Can be expanded as needed to make it more robust."""
    
    # Confirm that there is data
    if df.shape[0]<1:
        raise ValueError('No observations found')
        
    # Confirm right number of columns
    # Only some columns are used for modeling, so this can be modified
    # to look for specific columns. However, this way is more generic - 
    # no need to update function if modeling is updated to use other features
    if (not target_exists)+df.shape[1]!=12:
        raise ValueError('Incorrect number of features')

    # Check that all columns are named as expected
    # If target is present, data frame must be identified as training set
    for column in df:
        if column == 'Survived' and not target_exists:
            raise ValueError('Target variable in test data set')            
        elif column not in ['PassengerId','Survived','Pclass','Name','Sex',
                          'Age','SibSp','Parch','Ticket','Fare',
                          'Cabin','Embarked']:
            raise ValueError('Unexpected features')


# This module can be improved by making it less specific to this project.
# Errors are captured, but not handled - they are simply re-raised.
# With further development, errors should be logged.
