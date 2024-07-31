import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from functions import *


# this function will literally only import the black pit of despair that was my formatted flu data. IDK how it works. Will have to rewrite to format new data. 
def data_load_new(csv,nin, full = True, dropnan = True, norm_ = True):
      raw_data = pd.read_csv(csv)

      #make a dataset
      dataset = raw_data.copy()
      date = dataset['REGION']
      #drop the date
      dataset = dataset.drop(columns='REGION')
      data_stats = dataset.describe()
      data_stats = data_stats.transpose()
      
      #normalize the data using custom mini function => see functions.py
      normed_data = norm(dataset, data_stats, norm = norm_ )
      
      
      if full == True :
            #don't drop any data
            normed_data_1 = normed_data
      else:
            normed_data_1 = normed_data.drop(columns=['tempmax','tempmin','humidity','precip','windspeedmean','solarradiation','uvindex',])
            

      
      
     
      #1 week into future
      #convert to timeseries for learning (basically make stime steps to aid in training)
      #timelag
      #labels 
      nout=10
      series_data = series_to_supervised(normed_data_1, n_in = nin, n_out = nout, dropnan = dropnan)
      series_data.tail()
      
      if nout == 10 and full == True:
            series_data = series_data.drop(columns=['var1(t+1)',
                                                    'var2(t+1)', 'var4(t+1)', 'var5(t+1)', 'var6(t+1)',
                                                    'var7(t+1)', 'var8(t+1)', 'var9(t+1)', 'var10(t+1)', 'var11(t+1)',
                                                    'var1(t+2)', 'var2(t+2)',
                                                    'var4(t+2)', 'var5(t+2)', 'var6(t+2)', 'var7(t+2)', 'var8(t+2)',
                                                    'var9(t+2)', 'var10(t+2)', 'var11(t+2)',
                                                    'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                                    'var6(t+3)', 'var7(t+3)', 'var8(t+3)', 'var9(t+3)', 'var10(t+3)',
                                                    'var11(t+3)', 'var1(t+4)', 'var2(t+4)',
                                                    'var4(t+4)', 'var5(t+4)', 'var6(t+4)', 'var7(t+4)',
                                                    'var8(t+4)', 'var9(t+4)', 'var10(t+4)', 'var11(t+4)',
                                                    'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var6(t+5)','var7(t+5)','var8(t+5)',
                                                    'var9(t+5)','var10(t+5)','var11(t+5)','var1(t+6)','var2(t+6)',
                                                    'var4(t+6)','var5(t+6)','var6(t+6)','var7(t+6)','var8(t+6)','var9(t+6)','var10(t+6)',
                                                    'var11(t+6)', 'var1(t+7)',
                                                    'var2(t+7)','var4(t+7)','var5(t+7)','var6(t+7)','var7(t+7)','var8(t+7)','var9(t+7)','var10(t+7)',
                                                    'var11(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                                    'var6(t+8)','var7(t+8)','var8(t+8)','var9(t+8)','var10(t+8)','var11(t+8)',
                                                    'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)','var6(t+9)','var7(t+9)','var8(t+9)','var9(t+9)',
                                                    'var10(t+9)','var11(t+9)'])
      
      elif nout==10 and full == False:
            series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)','var1(t+2)', 'var2(t+2)',
                                              'var1(t+3)', 'var2(t+3)', 'var1(t+4)', 'var2(t+4)',
                                              'var1(t+5)','var2(t+5)','var1(t+6)','var2(t+6)',
                                              'var1(t+7)','var2(t+7)','var1(t+8)','var2(t+8)', 'var1(t+9)','var2(t+9)'])
      
     
      else:
            print('Error! Check your shitty code')
            
      return series_data


