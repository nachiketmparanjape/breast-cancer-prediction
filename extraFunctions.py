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
    