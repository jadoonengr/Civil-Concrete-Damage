# -*- coding: utf-8 -*-
# IMPORT LIBRARIES
import os
import scipy.io
import numpy as np
import pandas as pd
from math import log10, floor
from datetime import datetime

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb

from IPython.display import SVG
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# STEP 1: DATA GATHERING/CLEANING
# ----------------------------------------------------------------------------

seed = 7
np.random.seed(seed)

# IMPORT DATA FROM EXCEL
temp1 = pd.read_excel("Type 1.xlsx", sheetname="B1")
temp2 = pd.read_excel("Type 1.xlsx", sheetname="B2")
temp3 = pd.read_excel("Type 1.xlsx", sheetname="B3")

# SELECT RELEVANT COLUMNS 
train_set = temp1.iloc[:,0:20]
test_set = temp2.iloc[:,0:21]

# REMOVE INVALID ROWS
train_set = train_set[train_set.ID!='ID']
train_set.dropna(inplace=True)
test_set = test_set[test_set.ID!='ID']
test_set.dropna(inplace=True)

# RESET TABLE INDICES
train_set.reset_index(drop=True,inplace=True)
test_set.reset_index(drop=True,inplace=True)

# ENSURE TRAINING DATATYPES INTEGRITY
train_set['ID'] = train_set['ID'].astype(int)
train_set['DDD'] = train_set['DDD'].astype(int)
train_set['PARA1'] = train_set['PARA1'].astype(float)
train_set['CH'] = train_set['CH'].astype(int)
train_set['RISE'] = train_set['RISE'].astype(int)
train_set['COUN'] = train_set['COUN'].astype(int)
train_set['ENER'] = train_set['ENER'].astype(int)
train_set['DURATION'] = train_set['DURATION'].astype(int)
train_set['AMP'] = train_set['AMP'].astype(int)
train_set['A-FRQ'] = train_set['A-FRQ'].astype(int)
train_set['RMS'] = train_set['RMS'].astype(float)
train_set['ASL'] = train_set['ASL'].astype(int)
train_set['PCNTS'] = train_set['PCNTS'].astype(int)
train_set['THR'] = train_set['THR'].astype(int)
train_set['R-FRQ'] = train_set['R-FRQ'].astype(int)
train_set['I-FRQ'] = train_set['I-FRQ'].astype(int)
train_set['SIG-STRNGTH'] = train_set['SIG-STRNGTH'].astype(float)
train_set['ABS-ENERGY'] = train_set['ABS-ENERGY'].astype(float)
train_set['DAMAGE'] = train_set['DAMAGE'].astype(int)

# ENSURE TEST DATATYPES INTEGRITY
test_set['ID'] = test_set['ID'].astype(int)
test_set['DDD'] = test_set['DDD'].astype(int)
test_set['PARA1'] = test_set['PARA1'].astype(float)
test_set['CH'] = test_set['CH'].astype(int)
test_set['RISE'] = test_set['RISE'].astype(int)
test_set['COUN'] = test_set['COUN'].astype(int)
test_set['ENER'] = test_set['ENER'].astype(int)
test_set['DURATION'] = test_set['DURATION'].astype(int)
test_set['AMP'] = test_set['AMP'].astype(int)
test_set['A-FRQ'] = test_set['A-FRQ'].astype(int)
test_set['RMS'] = test_set['RMS'].astype(float)
test_set['ASL'] = test_set['ASL'].astype(int)
test_set['PCNTS'] = test_set['PCNTS'].astype(int)
test_set['THR'] = test_set['THR'].astype(int)
test_set['R-FRQ'] = test_set['R-FRQ'].astype(int)
test_set['I-FRQ'] = test_set['I-FRQ'].astype(int)
test_set['ABS-ENERGY'] = test_set['ABS-ENERGY'].astype(float)
test_set['DAMAGE'] = test_set['DAMAGE'].astype(int)

# MERGE SIG & STRNGTH COLUMNS (AN ISSUE IN TEST DATA)
test_set['SIG-STRNGTH'] = (test_set["SIG"].astype(str).map(str) + test_set["STRNGTH"].astype(str)).astype(float)
test_set.drop('SIG',axis=1,inplace=True)
test_set.drop('STRNGTH',axis=1,inplace=True)

# GENERATE TRAIN/TEST COLUMNS FOR TIME 
train_set['HR'] = train_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.hour)
train_set['MIN'] = train_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.minute)
train_set['SEC'] = train_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.second)
train_set['USEC'] = train_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.microsecond)

test_set['HR'] = test_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.hour)
test_set['MIN'] = test_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.minute)
test_set['SEC'] = test_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.second)
test_set['USEC'] = test_set['HH:MM:SS.mmmuuun'].apply(lambda x: x.microsecond)

# GENERATE TRAINING DATA (X,y), TEST DATA (X_test,y_test)
X = train_set.copy()
y = X.pop('DAMAGE')
X_test = test_set.copy()
y_test = X_test.pop('DAMAGE')

# DROP TIME COLUMN
X.drop('HH:MM:SS.mmmuuun',axis=1,inplace=True)
#X.drop('SIG-STRNGTH',axis=1,inplace=True)
X_test.drop('HH:MM:SS.mmmuuun',axis=1,inplace=True)
#X_test.drop('SIG-STRNGTH',axis=1,inplace=True)

# ARRANGE TRAIN/TEST COLUMNS SAME
X_test = X_test[X.columns]

# SPLIT TRAINING AND VALIDATION SETS (use scikit learn)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# DATA IS READY NOW
# TRAINING: X_train, y_train
# VALIDATION: X_valid, y_valid
# TEST: X_test, y_test


# ----------------------------------------------------------------------------
# STEP 2: MODEL TRAINING
# ----------------------------------------------------------------------------






# ----------------------------------------------------------------------------
# STEP 3: PREDICTION AND MODEL EVALUATION
# ----------------------------------------------------------------------------















