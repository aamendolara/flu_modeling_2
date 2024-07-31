import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from functions import *
from the_model import *


#function to predict the next segment of data, for forward chaining 
def basic_prediction_model(data, labels, test_data, test_labels, prediction_type, batch_ = 270, full=True):
    tf.keras.backend.clear_session()

    prometheus_week1to10 = build_modelE(data, labels)
    
    history = prometheus_week1to10.fit(data,
                                       labels,
                                       epochs=500,
                                       batch_size=batch_,
                                       validation_split=0.2,
                                       verbose=1,
                                       shuffle=False,
                                       callbacks=[reduce_lr, early_stop])
    
    history2 = prometheus_week1to10.fit(data,
                                       labels,
                                       epochs=500,
                                       batch_size=batch_,
                                       validation_split=0.2,
                                       verbose=1,
                                       shuffle=False,
                                       callbacks=[reduce_lr, early_stop])
    
    if prediction_type == 1:
          predictionsall2 = predict_future_t4(prometheus_week1to10, test_data, full=full)
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 2:
          predictionsall2 = predict_future_r10(prometheus_week1to10, test_data)
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 3:
          predictionsall2 = predict_future_10(prometheus_week1to10, test_data)
          predictionsall2 = predictionsall2.values
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 4:
          predictionsall2 = predict_future(prometheus_week1to10, test_data)
          error = 0
    elif prediction_type == 5:
          predictionsall2 = predict_future_r20(prometheus_week1to10, test_data)
          error = 0
    elif prediction_type == 0:
        return history2
    
     
    return [predictionsall2, error, history, history2, prometheus_week1to10]

#function to predict the next segment of data, for forward chaining, different model slightly
def basic_prediction_model2(data, labels, test_data, test_labels, prediction_type, batch_ = 270, full=True):
    tf.keras.backend.clear_session()

    prometheus_week1to10 = build_modelE(data, labels)
    
    history = prometheus_week1to10.fit(data,
                                       labels,
                                       epochs=500,
                                       batch_size=batch_,
                                       validation_split=0.2,
                                       verbose=1,
                                       shuffle=False,
                                       callbacks=[reduce_lr, early_stop])
    
    history2 = prometheus_week1to10.fit(data,
                                       labels,
                                       epochs=500,
                                       batch_size=batch_,
                                       validation_split=0.2,
                                       verbose=1,
                                       shuffle=False,
                                       callbacks=[reduce_lr, early_stop])
    
    if prediction_type == 1:
          predictionsall2 = predict_future_t4(prometheus_week1to10, test_data, full=full)
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 2:
          predictionsall2 = predict_future_r10(prometheus_week1to10, test_data)
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 3:
          predictionsall2 = predict_future_10(prometheus_week1to10, test_data)
          predictionsall2 = predictionsall2.values
          error = mae(predictionsall2, test_labels)
    elif prediction_type == 4:
          predictionsall2 = predict_future(prometheus_week1to10, test_data)
          error = 0
    elif prediction_type == 5:
          predictionsall2 = predict_future_r20(prometheus_week1to10, test_data)
          error = 0
    elif prediction_type == 0:
        return history2
    
     
    return [predictionsall2, error, history, history2, prometheus_week1to10]

#function for predicting 540 week data split
def threeway_split_model(data1,labels1, data2, labels2, test_data, test_labels, prediction_type, batch_ = 270, save_path = '',full=True, new_model = False, save_model = False):
      tf.keras.backend.clear_session()

      if new_model:
          prometheus_week1to10 = build_modelG(data1, labels1)
      else:
          prometheus_week1to10 = build_modelE(data1, labels1)

      history = prometheus_week1to10.fit(data1,
                                         labels1,
                                         epochs=500,
                                         batch_size=batch_,
                                         validation_split=0.2,
                                         verbose=1,
                                         shuffle=False,
                                         callbacks=[reduce_lr, early_stop])
      
      history2 = prometheus_week1to10.fit(data2,
                                          labels2,
                                          epochs=500,
                                          batch_size=batch_,
                                          validation_split=0.2,
                                          verbose=1,
                                          shuffle=False,
                                          callbacks=[reduce_lr, early_stop])
      
      if save_model == True:
          prometheus_week1to10.save(save_path)
      
      if prediction_type == 1:
            predictionsall2 = predict_future_t4(prometheus_week1to10, test_data, full=full)
            error = mae(predictionsall2, test_labels)
      elif prediction_type == 2:
            predictionsall2 = predict_future_r10(prometheus_week1to10, test_data)
            error = mae(predictionsall2, test_labels)
      elif prediction_type == 3:
            predictionsall2 = predict_future_10(prometheus_week1to10, test_data)
            predictionsall2 = predictionsall2.values
            error = mae(predictionsall2, test_labels)
      elif prediction_type == 4:
            predictionsall2 = predict_future(prometheus_week1to10, test_data)
            error = 0
      elif prediction_type == 5:
            predictionsall2 = predict_future_r20(prometheus_week1to10, test_data)
            error = 0
      elif prediction_type == 0:
          return history2
      
       
      return [predictionsall2, error, history, history2, prometheus_week1to10]

#function for predicting data set that has been split into training and test sets
def twoway_data(data1,labels1, test_data, test_labels, prediction_type, full=True, new_model = False):
      tf.keras.backend.clear_session()
      
      if new_model:
          prometheus_week1to10 = build_modelG(data1, labels1)
      else:
          prometheus_week1to10 = build_modelE(data1, labels1)

      prometheus_week1to10.fit(data1,
                               labels1,
                               epochs=500,
                               batch_size=270,
                               validation_split=0.2,
                               verbose=1,
                               shuffle=False,
                               callbacks=[reduce_lr, early_stop])
      
      
      if prediction_type == 1:
            predictionsall2 = predict_future_t4(prometheus_week1to10, test_data, full=full)
      elif prediction_type == 2:
            predictionsall2 = predict_future_r10(prometheus_week1to10, test_data)
      elif prediction_type == 3:
            predictionsall2 = predict_future_10(prometheus_week1to10, test_data)
            predictionsall2 = predictionsall2.values
            
            
      error = mae(predictionsall2, test_labels)
      
      return [predictionsall2, error]

#function for predicting data set split into 1 training set and 2 prediction sets, error for predictions is combined 
def twoway_prediction_model(data, labels, test1, test_label1, test2, testlabel2, prediction_type, full=True):
      tf.keras.backend.clear_session()

      prometheus_week1to10 = build_modelE(data, labels)

      prometheus_week1to10.fit(data,
                               labels,
                               epochs=500,
                               batch_size=270,
                               validation_split=0.2,
                               verbose=1,
                               shuffle=False,
                               callbacks=[reduce_lr, early_stop])
      
      
      if prediction_type == 1:
            predictionsall2 = predict_future_t4(prometheus_week1to10, test1, full=full)
            predictionsall = predict_future_t4(prometheus_week1to10, test2, full=full)
      elif prediction_type == 2:
            predictionsall2 = predict_future_r10(prometheus_week1to10, test1)
            predictionsall = predict_future_r10(prometheus_week1to10, test2)
      elif prediction_type == 3:
            predictionsall2 = predict_future_10(prometheus_week1to10, test1)
            predictionsall = predict_future_10(prometheus_week1to10, test2)
            predictionsall2 = predictionsall2.values
            predictionsall = predictionsall.values
            
      error = mae(predictionsall2, test_label1)
      error1 = mae(predictionsall, testlabel2)
      print(error)
      
      error_comby = np.zeros([2,3])
      error_comby[0,0] = (error.loc['mean','Prediction +1 Error'] + error1.loc['mean','Prediction +1 Error'])/2
      error_comby[0,1] = (error.loc['mean','Prediction +5 Error'] + error1.loc['mean','Prediction +5 Error'])/2
      error_comby[0,2] = (error.loc['mean','Prediction +10 Error'] + error1.loc['mean','Prediction +10 Error'])/2
      error_comby[1,0] = ((error.loc['std','Prediction +1 Error']**2 + error1.loc['std','Prediction +1 Error']**2)/2)**.5
      error_comby[1,1] = ((error.loc['std','Prediction +5 Error']**2 + error1.loc['std','Prediction +5 Error']**2)/2)**.5
      error_comby[1,2] = ((error.loc['std','Prediction +10 Error']**2 + error1.loc['std','Prediction +10 Error']**2)/2)**.5
      
      return [predictionsall2, error_comby]

# def forward_chain_validation(n, data):
#     time_steps = data.shape[1]
#     n_val = time_steps//n
    
#     for i in range(n):

#         data_1_1 = pd.DataFrame(data=data.iloc[0:80])
#         data_1_2 = pd.DataFrame(data=data.iloc[80:160])
#         [data1,data2,labels1,labels2,labels10,labels20] = data_maker2(data_1_1, data_1_2)
#         model_1 = basic_prediction_model(data1, labels1, data2, labels10, prediction_type=1, batch_= 270, full= False)       