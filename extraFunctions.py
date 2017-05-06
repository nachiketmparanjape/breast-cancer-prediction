"""

This files contains custom functions designed for data exploration and machine learning

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Create training and test sets
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from sklearn.model_selection import KFold, cross_val_score # Cross validation

def class_balance(classcolumnSeries):
    
    """
    This function takes in a Pandas Series object with the classification variable (binary)
    Returns percent population of each category
    """
    
    total_size_of_dataset = float(len(classcolumnSeries))
    
    i = 1
    for item in list(classcolumnSeries.value_counts().index):
        
        pct = round(float(classcolumnSeries.value_counts()[item]) * 100.0 / total_size_of_dataset)
        print (str(i) + ") " + item + " - " + str(pct) + " %")
        i += 1
    
    
def normalize_df(frame):
    '''
    Helper function to Normalize data set
    Intializes an empty data frame which will normalize all floats types
    and just append the non-float types so basically the class in our data frame
    '''
    breastCancerNorm = pd.DataFrame()
    for item in frame:
        if item in frame.select_dtypes(include=[np.float]):
            breastCancerNorm[item] = ((frame[item] - frame[item].min()) / 
                            (frame[item].max() - frame[item].min()))
        else:
            breastCancerNorm[item] = frame[item]
   
    return breastCancerNorm