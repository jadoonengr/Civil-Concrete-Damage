# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 03:12:39 2017

@author: Aamir
"""

# Import necessary modules
import os
import scipy.io
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# Import data
os.chdir("c:/Users/Inspiron/Desktop/Deep Learning/civil")
data = pd.read_excel("1.xlsx")
data = data.iloc[:,[4, 5, 6, 7, 8, 9, 10, 17, 18, 19]]
data = data.dropna()

d = np.array(data.iloc[:,0:9])
predictors = (d - d.mean())/d.std()
target = np.eye(4)[data['DAMAGE'].astype(int)]
n_cols = predictors.shape[1]
    
# Specify the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape = (n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))

early_stopping_monitor = EarlyStopping(patience=2)

# Compile the model
sgd = SGD(lr=0.1)#, decay=1e-6, momentum=1.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target, validation_split=0.3, batch_size=64, epochs=100, callbacks=[early_stopping_monitor])

# Save the model
#model.save('model_file.h5')
#mm = load_model('model_file.h5')
#predictions = mm.predict(mnist.validation.images)
#prob_true = predictions[:,1]