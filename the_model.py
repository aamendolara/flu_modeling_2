import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from functions import *

#make a simple model 
#rejected
def build_modelA():
    model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=[len(normed_training.keys())]),
            layers.Dropout(.5),
            layers.Dense(16, activation='relu'),
            layers.Dropout(.2),
            layers.Dense(8, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(1)
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return model 



#make a simple model
#rejected
def build_modelB():
    model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(normed_training.keys())]),
            layers.Dropout(.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(.2),
            layers.Dense(8, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(1)
        ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return model 



    
#probably what is should have done to begin with
#rejected
def build_modelC(data):
    model = keras.Sequential([
                  layers.Dense(32, activation='selu', input_shape=[len(data.keys())]),
                  layers.Dropout(0.2),
                  layers.Dense(16, activation='selu'),
                  layers.Dense(1)
         ])
    
    optimizer = tf.keras.optimizers.Nadam()
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return model 

#trying LSTM baseline 
def build_modelD(data, labels):
      samples = data.shape[0]
      time_steps = data.shape[1]
      features = data.shape[2]
      if len(labels.shape) == 2:
            label_shape = labels.shape[1]
      else:
            label_shape = 1
      
      model = keras.Sequential([
                  layers.LSTM(50, batch_input_shape=((1,time_steps,features)), return_sequences=True),
                  layers.Dense(1)
      ])
      
      model.compile(loss='mse',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['mae','mse'])
      return model

#best so far using 1/3 of the data for training
#able to make good predictions
#switched to mae for loss function
#low dropout seems to help
#BEST possibly final 
#works reasonably well regardless of which section of data used to train
#THIS MODEL LOOKS 1 or 2 WEEKS INTO THE FUTURE well
#will also produce some recursive predictions. Not too bad considering its looking years ahead -- room for improvement

def build_modelE(data, labels):
      #take sample size for input. currently set to one 
      samples = data.shape[0]
      # time steps and features taken from data shape to simplify building model
      time_steps = data.shape[1]
      features = data.shape[2]
      #output shape
      if len(labels.shape) == 2:
            label_shape = labels.shape[1]
      else:
            label_shape = 1
      
      
      model = keras.Sequential([
                  keras.layers.Bidirectional(layers.LSTM(500, return_sequences=True, batch_input_shape=((samples,time_steps,features))), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(500, dropout=.5, return_sequences=True), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(500, dropout=.5),merge_mode='concat'),
                  keras.layers.Dense(label_shape)
                  ])
      
      
      #uses mean absolute error as loss function
      model.compile(loss='mae',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['mae','mse', R_square])
      return model
  
def build_modelE2(data, labels):
      #take sample size for input. currently set to one 
      samples = data.shape[0]
      # time steps and features taken from data shape to simplify building model
      time_steps = data.shape[1]
      features = data.shape[2]
      #output shape
      if len(labels.shape) == 2:
            label_shape = labels.shape[1]
      else:
            label_shape = 1
      
      
      model = keras.Sequential([
                  keras.layers.Bidirectional(layers.LSTM(500, return_sequences=True, batch_input_shape=((samples,time_steps,features))), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(500, dropout=.5, return_sequences=True), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(500, dropout=.5),merge_mode='concat'),
                  keras.layers.Dense(label_shape)
                  ])
      
      
      #uses mean absolute error as loss function
      model.compile(loss='mae',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['mae','mse', R_square])
      return model
      
  

###taking a bit more of a regular approach to structure. Maybe will return and update earlier data? Idk. This may not actually perform better. We shall see. 
def build_modelG(data, labels):
      #take sample size for input. currently set to one 
      samples = data.shape[0]
      # time steps and features taken from data shape to simplify building model
      time_steps = data.shape[1]
      features = data.shape[2]
      #output shape
      if len(labels.shape) == 2:
            label_shape = labels.shape[1]
      else:
            label_shape = 1
      
        
      hidden_num = int((time_steps + label_shape) / 2)
      
      model = keras.Sequential([
                  keras.layers.Bidirectional(layers.LSTM(time_steps, return_sequences=True, batch_input_shape=((samples,time_steps,features))), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(hidden_num, dropout=.25, return_sequences=True), merge_mode='concat'),
                  keras.layers.Bidirectional(layers.LSTM(hidden_num, dropout=.25),merge_mode='concat'),
                  keras.layers.Dense(label_shape)
                  ])
      
      
      #uses mean absolute error as loss function
      model.compile(loss='mae',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['mae','mse', R_square])
      return model


# good model 
# step 2. produced usable lstm model
#see above for updated
#def build_modelF(data, labels):
#      samples = data.shape[0]
#      time_steps = data.shape[1]
#      features = data.shape[2]
#      if len(labels.shape) == 2:
#            label_shape = labels.shape[1]
#      else:
#            label_shape = 1
#      
#      
#       model = keras.Sequential([
#                   layers.LSTM(500, return_sequences=True, batch_input_shape=((samples,time_steps,features))),
#                   layers.LSTM(500, dropout=.25, return_sequences=True),
#                   layers.LSTM(500, dropout=.25),
#                   keras.layers.Dense(label_shape)
#                   ])
#      
#      model.compile(loss='mae',
#                    optimizer=tf.keras.optimizers.Adam(),
#                    metrics=['mae','mse',R_square])
#      return model
