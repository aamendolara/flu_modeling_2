import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend


#plot history 
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [% ILI]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$ILI^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.legend()
  plt.show()


#make the dot thing
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
#callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience= 200, restore_best_weights = True)


#save_state = keras.callbacks.ModelCheckpoint('C:/Users/eric_fortune/Desktop/THESIS Backup/Model_Weights/prometheusplus2.{epoch:02d}-{val_loss:.2f}.h5', interval = 1000)


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5,
                              patience=10, min_lr=0.000000001)

#normalize/standardize data
def norm(x, stats, norm = True):
    
    if norm == True:
        y = x.astype('float64')
        stats = stats.astype('float64')
        df = (y - stats['mean']) / stats['std']
        return df
    else:
        y = x.astype('float64')
        stats = stats.astype('float64')
        df = (y - stats['min']) / (stats['max'] - stats['min'])
        return df


#produce scatterplot comparing labels to predicted values
def scatterplot(labels, predictions):
    
    plt.scatter(labels, predictions)
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()
    
#plot predictions vs. labels
def plot_results(predictions, labels):
    predictions_df = pd.DataFrame(data=predictions, index=labels.index)
    
    plt.plot(predictions_df, color= 'red', label = 'Predicted ILI')
    plt.plot(labels, color='green', label = 'Actual ILI')
    plt.title('Predictions Overlayed Onto True Data')
    plt.xlabel('Week')
    plt.ylabel('% ILI')
    plt.legend()
    plt.show()

#line predictions up to plot nest to labels
def shift_predict(predictions, labels):
    predictions_df = pd.DataFrame(data=predictions, index=labels.index)
    
    return predictions_df

#combine several functions to produce a series of graphs evaluating prediction performance 
def analyze_model(model, history, training_data, training_labels, testing_data, testing_labels): 
       
    if not isinstance(training_labels, pd.DataFrame):
          training_labels = pd.DataFrame(data=training_labels)
          
          
    if not isinstance(testing_labels, pd.DataFrame):
          testing_labels = pd.DataFrame(data=testing_labels)
          
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    test_predictions = model.predict(testing_data).flatten()
    test_predictions = pd.DataFrame(data=test_predictions, index=testing_labels.index)
    
    
    plot_history(history)
    scatterplot(testing_labels, test_predictions)
    plt.plot(test_predictions, color= 'red', label = 'Predicted ILI')
    plt.plot(testing_labels, color='green', label = 'Actual ILI')
    plt.title('Predictions Overlayed Onto True Data')
    plt.xlabel('Week')
    plt.ylabel('% ILI')
    plt.legend()
    
    
    plt.show()

def plot_error(predictions, labels):
      error = predictions - labels
      plt.hist(error, bins = 25)
      plt.xlabel("Prediction Error")
      _ = plt.ylabel("Count")
      _ = plt.xlim(-7,7)
        
      plt.show()
    


def reshape_data(data):
      data = data.values
      data = data.reshape((data.shape[0],1, data.shape[1]))
      
      return data

def reshape_data_4D(data, num_seq):
      data = data.values
      data = data.reshape((data.shape[0],num_seq,1, data.shape[1]))
      
      
      return data
      
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def predict_future(model, data):
      
      #initialize a numpy array of zeros to store predictions
      predictions = np.zeros((data.shape[0],1))
      #create our editable data so we dont mess up the original 
      edit_data = data.copy()
      
      #loop that iterates predictions 
      for i in range(0,data.shape[0]):
            #start with a 2 week slice of data
            #grow as more is predicted
            base_week = edit_data[:i+1]
            #generate prediction on data 
            predict_week = model.predict(base_week)
            
            #add prediction to prediction list 
            predictions[i,0] = predict_week[-1,0]
            
      predictions = predictions.flatten()
      predictions = pd.DataFrame(data=predictions)
            
      return predictions 

def predict_future_r20(model, data):
      #recursive prediction by week
      #model will predict 2 weeks starting from 0, then add/replace the valuse in chart with predictions, then predict based
      #on those predictions. Will output array of predictions that should be independant of the "true" data for the prediction period
      #this particular function will predict until 10 weeks
      
      #initialize numpy array to hold predictions
      full_predictions = np.empty((data.shape[0],20))
      
      
      for j in range(0,data.shape[0]):
            
            
            #initialize a numpy array of zeros to store predictions
            predictions = np.zeros((20,1))
            
            #create our editable data so we dont mess up the original 
            edit_data = data.copy()
            
            #loop that iterates predictions 
            
            for i in range(j,j+20):
                  #start with a 2 week slice of data
                  #grow as more is predicted
                  base_week = edit_data[:i+1]
                  #generate prediction on data 
                  predict_week = model.predict(base_week)
                  
                  #loop prediction back into data
                  #[row, 3rd dimension == 0 , column]
                  
                  if i+1<data.shape[0]:
                        edit_data[i+1,0,2]=predict_week[-1]
                        
                        
                  
                  #add prediction to prediction list 
                  predictions[i-j,] = predict_week[-1,]
                  
                  
            predictions = predictions.flatten()
            predictions = np.reshape(predictions, (1,20))
            
            percent = round((j/data.shape[0])*100,2)
            print("{percent}%".format(percent=percent), end= " ", flush=True) 
            full_predictions[j,] = predictions[0,]
      
      
      return full_predictions 
            


def predict_future_10(model, data):
      #non-regressive prediction by 2 weeks
      #model will predict 2 weeks starting from 0, then add/replace the values in chart with predictions
     
      
      #initialize a numpy array of zeros to store predictions
      predictions = np.zeros((data.shape[0],10))
      #create our editable data so we dont mess up the original 
      edit_data = data.copy()
      
      #loop that iterates predictions 
      for i in range(0,data.shape[0]):
            #start with a 2 week slice of data
            #grow as more is predicted
            base_week = edit_data[:i+1]
            #generate prediction on data 
            predict_week = model.predict(base_week)
       
            
            #add prediction to prediction list 
            predictions[i,0], predictions[i,1], predictions[i,2], predictions[i,3], predictions[i,4], predictions[i,5], predictions[i,6], predictions[i,7], predictions[i,8], predictions[i,9] = predict_week[-1,0], predict_week[-1,1], predict_week[-1,2], predict_week[-1,3], predict_week[-1,4], predict_week[-1,5], predict_week[-1,6], predict_week[-1,7], predict_week[-1,8], predict_week[-1,9]
            
            
      predictions = pd.DataFrame(data=(predictions[:,0], predictions[:,1], predictions[:,2], predictions[:,3], predictions[:,4], predictions[:,5], predictions[:,6], predictions[:,7], predictions[:,8], predictions[:,9]))
      predictions = predictions.transpose()
            
      return predictions 


def predict_future_r10(model, data):
      #recursive prediction by week
      #model will predict 2 weeks starting from 0, then add/replace the valuse in chart with predictions, then predict based
      #on those predictions. Will output array of predictions that should be independant of the "true" data for the prediction period
      #this particular function will predict until 10 weeks
      
      #initialize numpy array to hold predictions
      full_predictions = np.empty((data.shape[0],10))
      
      
      for j in range(0,data.shape[0]):
            
            
            #initialize a numpy array of zeros to store predictions
            predictions = np.zeros((10,1))
            
            #create our editable data so we dont mess up the original 
            edit_data = data.copy()
            
            #loop that iterates predictions 
            
            for i in range(j,j+10):
                  #start with a 2 week slice of data
                  #grow as more is predicted
                  base_week = edit_data[:i+1]
                  #generate prediction on data 
                  predict_week = model.predict(base_week)
                  
                  #loop prediction back into data
                  #[row, 3rd dimension == 0 , column]
                  
                  if i+1<data.shape[0]:
                        edit_data[i+1,0,2]=predict_week[-1]
                        
                        
                  
                  #add prediction to prediction list 
                  predictions[i-j,] = predict_week[-1,]
                  
                  
            predictions = predictions.flatten()
            predictions = np.reshape(predictions, (1,10))
            
            percent = round((j/data.shape[0])*100,2)
            print("{percent}%".format(percent=percent), end= " ", flush=True) 
            full_predictions[j,] = predictions[0,]
      
      
      return full_predictions 
            
def predict_future_t4(model, data, full = True):
      #recursive prediction by week with 4 week lag
      #model will predict 2 weeks starting from 0, then add/replace the values in chart with predictions, then predict based
      #on those predictions. Will output array of predictions that should be independant of the "true" data for the prediction period
      #this particular function will predict until 10 weeks
      
      #initialize numpy array to hold predictions
      full_predictions = np.empty((data.shape[0],10))
      
      
      for j in range(0,data.shape[0]):
            
            
            #initialize a numpy array of zeros to store predictions
            predictions = np.zeros((10,1))
            
            #create our editable data so we dont mess up the original 
            edit_data = data.copy()
            
            #loop that iterates predictions 
            
            for i in range(j,j+10):
                  #start with a 2 week slice of data
                  #grow as more is predicted
                  base_week = edit_data[:i+1]
                  #generate prediction on data 
                  predict_week = model.predict(base_week)
                  
                  #loop predictions back into data
                  #[row, 3rd dimension == 0 , column]
                  if full==False:
                        #t-1
                        if i+1<data.shape[0]:
                              edit_data[i+1,0,20]=predict_week[-1]
                        #t-2
                        if i+2<data.shape[0] & i-j>=1:
                              edit_data[i+2,0,14]=predict_week[-1]
                        #t-3
                        if i+3<data.shape[0] & i-j>=2:
                              edit_data[i+3,0,8]=predict_week[-1]
                        #t-4
                        if i+4<data.shape[0] & i-j>=3:
                              edit_data[i+4,0,2]=predict_week[-1]
                  else:
                         #t-1
                        if i+1<data.shape[0]:
                              edit_data[i+1,0,41]=predict_week[-1]
                        #t-2
                        if i+2<data.shape[0] & i-j>=1:
                              edit_data[i+2,0,28]=predict_week[-1]
                        #t-3
                        if i+3<data.shape[0] & i-j>=2:
                              edit_data[i+3,0,15]=predict_week[-1]
                        #t-4
                        if i+4<data.shape[0] & i-j>=3:
                              edit_data[i+4,0,2]=predict_week[-1]
                  #add prediction to prediction list 
                  predictions[i-j,] = predict_week[-1,]

                        
            predictions = predictions.flatten()
            predictions = np.reshape(predictions, (1,10))
            
            j_number = j+1
            print("{j_num}/{number}".format(j_num=j_number,number=data.shape[0]), end= " ", flush=True) 
            full_predictions[j,] = predictions[0,]
      
      
      return full_predictions 
  
    
            
def R_square(y_true, y_pred):
      SS_res = keras.backend.sum(keras.backend.square(y_true-y_pred))
      SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
      return (1 - SS_res/(SS_tot + keras.backend.epsilon()))
    
      
#extract error and std dev of error for all weeks 
def mae(predictions, labels, return_error = False):
     
      
      error3 = pd.DataFrame(data=[labels[:,0], labels[:,4], labels[:,9], predictions[:,0], predictions[:,4], predictions[:,9]])
      error3 = error3.transpose()
      error3.columns = ['Label Data 1', 'Label Data 5', 'Label Data 10', 'Predictions +1 Week', 'Predictions +5 Weeks', 'Predictions +10 Weeks']
      error3['Prediction +1 Error']= abs(error3['Label Data 1']-error3['Predictions +1 Week'])
      error3['Prediction +5 Error']= abs(error3['Label Data 5']-error3['Predictions +5 Weeks'])
      error3['Prediction +10 Error']= abs(error3['Label Data 10']-error3['Predictions +10 Weeks'])
      disc = error3.describe()
      disc = disc.iloc[[1,2],[6,7,8]]
      
      if return_error == True:
            return [disc, error3]
      else:
            return disc
        

def data_maker(data_1_1, data_2_1, data_3_1):
      
      labels_1_1 = data_1_1.pop('var3(t)')
      labels_2_1 = data_2_1.pop('var3(t)')
      labels_3_1 = data_3_1.pop('var3(t)')
      
      labels_1_2 = data_1_1.pop('var3(t+1)')
      labels_2_2 = data_2_1.pop('var3(t+1)')
      labels_3_2 = data_3_1.pop('var3(t+1)')
      
      labels_1_3 = data_1_1.pop('var3(t+2)')
      labels_2_3 = data_2_1.pop('var3(t+2)')
      labels_3_3 = data_3_1.pop('var3(t+2)')
      
      labels_1_4 = data_1_1.pop('var3(t+3)')
      labels_2_4 = data_2_1.pop('var3(t+3)')
      labels_3_4 = data_3_1.pop('var3(t+3)')
      
      labels_1_5 = data_1_1.pop('var3(t+4)')
      labels_2_5 = data_2_1.pop('var3(t+4)')
      labels_3_5 = data_3_1.pop('var3(t+4)')
      
      labels_1_6 = data_1_1.pop('var3(t+5)')
      labels_2_6 = data_2_1.pop('var3(t+5)')
      labels_3_6 = data_3_1.pop('var3(t+5)')
      
      labels_1_7 = data_1_1.pop('var3(t+6)')
      labels_2_7 = data_2_1.pop('var3(t+6)')
      labels_3_7 = data_3_1.pop('var3(t+6)')
      
      labels_1_8 = data_1_1.pop('var3(t+7)')
      labels_2_8 = data_2_1.pop('var3(t+7)')
      labels_3_8 = data_3_1.pop('var3(t+7)')
      
      labels_1_9 = data_1_1.pop('var3(t+8)')
      labels_2_9 = data_2_1.pop('var3(t+8)')
      labels_3_9 = data_3_1.pop('var3(t+8)')
      
      labels_1_10 = data_1_1.pop('var3(t+9)')
      labels_2_10 = data_2_1.pop('var3(t+9)')
      labels_3_10 = data_3_1.pop('var3(t+9)')
      
      
      #combine for later use
      labels_1_10 = pd.concat([labels_1_1, labels_1_2, labels_1_3, labels_1_4, labels_1_5, labels_1_6, labels_1_7, labels_1_8, labels_1_9, labels_1_10], axis=1)
      labels_2_10 = pd.concat([labels_2_1, labels_2_2, labels_2_3, labels_2_4, labels_2_5, labels_2_6, labels_2_7, labels_2_8, labels_2_9, labels_2_10], axis=1)
      labels_3_10 = pd.concat([labels_3_1, labels_3_2, labels_3_3, labels_3_4, labels_3_5, labels_3_6, labels_3_7, labels_3_8, labels_3_9, labels_3_10], axis=1)

      labels_1_10 = labels_1_10.to_numpy()
      labels_2_10 = labels_2_10.to_numpy()
      labels_3_10 = labels_3_10.to_numpy()
      
      
      #data_2_ns = reshape_data(data_2_ns)
      #reshaoe labels to 3D 

      labels_1_1 = labels_1_1.to_numpy()
      labels_2_1 = labels_2_1.to_numpy()
      labels_3_1 = labels_3_1.to_numpy()
      
      data_1_1 = reshape_data(data_1_1)
      data_2_1 = reshape_data(data_2_1)
      data_3_1 = reshape_data(data_3_1)
            
      return [data_1_1, data_2_1, data_3_1, labels_1_1, labels_2_1, labels_3_1, labels_1_10, labels_2_10, labels_3_10]

def data_maker2(data_1_1, data_2_1):
      
      labels_1_1 = data_1_1.pop('var3(t)')
      labels_2_1 = data_2_1.pop('var3(t)')
     
      
      labels_1_2 = data_1_1.pop('var3(t+1)')
      labels_2_2 = data_2_1.pop('var3(t+1)')
   
      
      labels_1_3 = data_1_1.pop('var3(t+2)')
      labels_2_3 = data_2_1.pop('var3(t+2)')
   
      
      labels_1_4 = data_1_1.pop('var3(t+3)')
      labels_2_4 = data_2_1.pop('var3(t+3)')

      
      labels_1_5 = data_1_1.pop('var3(t+4)')
      labels_2_5 = data_2_1.pop('var3(t+4)')
      
      labels_1_6 = data_1_1.pop('var3(t+5)')
      labels_2_6 = data_2_1.pop('var3(t+5)')
      
      
      labels_1_7 = data_1_1.pop('var3(t+6)')
      labels_2_7 = data_2_1.pop('var3(t+6)')
    
      
      labels_1_8 = data_1_1.pop('var3(t+7)')
      labels_2_8 = data_2_1.pop('var3(t+7)')
  
      
      labels_1_9 = data_1_1.pop('var3(t+8)')
      labels_2_9 = data_2_1.pop('var3(t+8)')
   
      
      labels_1_10 = data_1_1.pop('var3(t+9)')
      labels_2_10 = data_2_1.pop('var3(t+9)')

      
      
      #combine for later use
      labels_1_10 = pd.concat([labels_1_1, labels_1_2, labels_1_3, labels_1_4, labels_1_5, labels_1_6, labels_1_7, labels_1_8, labels_1_9, labels_1_10], axis=1)
      labels_2_10 = pd.concat([labels_2_1, labels_2_2, labels_2_3, labels_2_4, labels_2_5, labels_2_6, labels_2_7, labels_2_8, labels_2_9, labels_2_10], axis=1)
      

      labels_1_10 = labels_1_10.to_numpy()
      labels_2_10 = labels_2_10.to_numpy()
     
      
      #data_2_ns = reshape_data(data_2_ns)
      #reshaoe labels to 3D 

      labels_1_1 = labels_1_1.to_numpy()
      labels_2_1 = labels_2_1.to_numpy()
      
      
      data_1_1 = reshape_data(data_1_1)
      data_2_1 = reshape_data(data_2_1)
      
            
      return [data_1_1, data_2_1, labels_1_1, labels_2_1, labels_1_10, labels_2_10]

def data_maker1(data_1_1):
      
      labels_1_1 = data_1_1.pop('var3(t)')
      
      
      labels_1_2 = data_1_1.pop('var3(t+1)')
      
      
      labels_1_3 = data_1_1.pop('var3(t+2)')
      
      
      labels_1_4 = data_1_1.pop('var3(t+3)')
      
      
      labels_1_5 = data_1_1.pop('var3(t+4)')
      
      
      labels_1_6 = data_1_1.pop('var3(t+5)')
      
      
      labels_1_7 = data_1_1.pop('var3(t+6)')
        
      
      labels_1_8 = data_1_1.pop('var3(t+7)')
      
      
      labels_1_9 = data_1_1.pop('var3(t+8)')
        
      
      labels_1_10 = data_1_1.pop('var3(t+9)')
      

      
      
      #combine for later use
      labels_1_10 = pd.concat([labels_1_1, labels_1_2, labels_1_3, labels_1_4, labels_1_5, labels_1_6, labels_1_7, labels_1_8, labels_1_9, labels_1_10], axis=1)
      

      labels_1_10 = labels_1_10.to_numpy()
     
      
      #data_2_ns = reshape_data(data_2_ns)
      #reshape labels to 3D 

      labels_1_1 = labels_1_1.to_numpy()
      
      
      data_1_1 = reshape_data(data_1_1)
      
            
      return [data_1_1, labels_1_1, labels_1_10]


def sim_data_load(csv,nin, futuret = True, offset_T = False, norm_ = True, dropnan = True, empty = False):
    raw_data = pd.read_csv(csv)
    dataset = raw_data.copy()
    data_stats = dataset.describe()
    data_stats = data_stats.transpose()
    
    #normalize the data using custom mini function => see functions.py
    normed_data = norm(dataset, data_stats, norm = norm_ )
    
    if offset_T==False:
        normed_data = normed_data.drop(columns='Temp_off')
        
    else:
        normed_data = normed_data.drop(columns='Temp')
    
    #1 week into future
    #convert to timeseries for learning (basically make stime steps to aid in training)
    #timelag
    #labels 
    nout=10
    series_data = series_to_supervised(normed_data, n_in = nin, n_out = nout, dropnan = dropnan)
    series_data.tail()
    
    if nout==10 and futuret==True:
          series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)', 'var4(t+1)', 'var5(t+1)', 'var6(t+1)','var1(t+2)', 'var2(t+2)',
                                            'var4(t+2)', 'var5(t+2)', 'var6(t+2)',
                                            'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                            'var6(t+3)', 'var1(t+4)', 'var2(t+4)',
                                            'var4(t+4)', 'var5(t+4)', 'var6(t+4)',
                                            'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var6(t+5)','var1(t+6)','var2(t+6)',
                                            'var4(t+6)','var5(t+6)','var6(t+6)', 'var1(t+7)',
                                            'var2(t+7)','var4(t+7)','var5(t+7)','var6(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                            'var6(t+8)',
                                            'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)','var6(t+9)'])
    
    elif nout==10 and futuret==False:
          series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)', 'var4(t+1)', 'var5(t+1)','var1(t+2)', 'var2(t+2)',
                                            'var4(t+2)', 'var5(t+2)',
                                            'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                            'var1(t+4)', 'var2(t+4)',
                                            'var4(t+4)', 'var5(t+4)',
                                            'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var1(t+6)','var2(t+6)',
                                            'var4(t+6)','var5(t+6)','var1(t+7)',
                                            'var2(t+7)','var4(t+7)','var5(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                            'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)',])
          
    elif nout==10 and futuret==False and empty == True:
          series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)','var1(t+2)', 'var2(t+2)',
                                            'var1(t+3)', 'var2(t+3)', 
                                            'var1(t+4)', 'var2(t+4)',
                                            'var1(t+5)','var2(t+5)','var1(t+6)','var2(t+6)',
                                            'var1(t+7)',
                                            'var2(t+7)','var1(t+8)','var2(t+8)',
                                            'var1(t+9)','var2(t+9)',])
         
            
    return series_data


# this function will literally only import the black pit of despair that was my formatted flu data. IDK how it works. Will have to rewrite to format new data. 
def data_load(csv,nin, futuret = True, full = True, dropnan = True, outliers = True, norm_ = True, empty = False):
      raw_data = pd.read_csv(csv)

      #make a dataset
      dataset = raw_data.copy()
      date = dataset['DATE']
      #drop the date
      dataset = dataset.drop(columns='DATE')
      data_stats = dataset.describe()
      data_stats = data_stats.transpose()
      
      #normalize the data using custom mini function => see functions.py
      normed_data = norm(dataset, data_stats, norm = norm_ )
      
      if outliers==False:
           normed_data = normed_data.drop(range(0,52), axis=0)
           normed_data = normed_data.drop(range(290,342), axis=0)
           normed_data = normed_data.drop(range(720,772), axis=0)
           normed_data = normed_data.reset_index(drop=True)
            
      
      if full == True :
            #drop some data, lets build from the ground up everything removed is either not useful or not usable
            normed_data_1 = normed_data.drop(columns=['ABOVE BASELINE','IS FLUWEEK', 'AGE 0-4', 'AGE 5-24', 'AGE 25-64', 
                                                      'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS',
                                                      'TOTAL SPECIMENS', 'A (Subtyping not Performed)', 'A (2009 H1N1)',
                                                      'A (H1)', 'A (H3)', 'B', 'Bvic', 'Byam','COOLING DAYS MONTHLY', 'HEATING DAYS MONTHLY'])
      else:
            normed_data_1 = normed_data.drop(columns=['ABOVE BASELINE','IS FLUWEEK', 'AGE 0-4', 'AGE 5-24', 'AGE 25-64', 
                                 'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS',
                                 'TOTAL SPECIMENS', 'A (Subtyping not Performed)', 'A (2009 H1N1)',
                                 'A (H1)', 'A (H3)', 'B', 'Bvic', 'Byam','COOLING DAYS MONTHLY', 'HEATING DAYS MONTHLY','AVG WIND SPEED MONTHLY', 'PRECIPITATION MONTHLY', 'AVG TEMP MONTHLY',
                                 'POPULATION', 'VAXEFFECTIVENESS', 'VACCINERATENATIONAL', 'AWND',])
            
      if empty == True:
            normed_data_1 = normed_data.drop(columns=['ABOVE BASELINE','IS FLUWEEK', 'AGE 0-4', 'AGE 5-24', 'AGE 25-64', 
                               'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS',
                               'TOTAL SPECIMENS', 'A (Subtyping not Performed)', 'A (2009 H1N1)',
                               'A (H1)', 'A (H3)', 'B', 'Bvic', 'Byam','COOLING DAYS MONTHLY', 'HEATING DAYS MONTHLY','AVG WIND SPEED MONTHLY', 'PRECIPITATION MONTHLY', 'AVG TEMP MONTHLY',
                               'POPULATION', 'VAXEFFECTIVENESS', 'VACCINERATENATIONAL', 'AWND','PRCP','TAVG'])
          
      
      
     
      #1 week into future
      #convert to timeseries for learning (basically make stime steps to aid in training)
      #timelag
      #labels 
      nout=10
      series_data = series_to_supervised(normed_data_1, n_in = nin, n_out = nout, dropnan = dropnan)
      series_data.tail()
      
      if nout == 10 and futuret == True and full == True:
            series_data = series_data.drop(columns=['var1(t+1)',
                                                    'var2(t+1)', 'var4(t+1)', 'var5(t+1)', 'var6(t+1)',
                                                    'var7(t+1)', 'var8(t+1)', 'var9(t+1)', 'var10(t+1)', 'var11(t+1)',
                                                    'var12(t+1)','var1(t+2)', 'var2(t+2)',
                                                    'var4(t+2)', 'var5(t+2)', 'var6(t+2)', 'var7(t+2)', 'var8(t+2)',
                                                    'var9(t+2)', 'var10(t+2)', 'var11(t+2)', 'var12(t+2)',
                                                    'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                                    'var6(t+3)', 'var7(t+3)', 'var8(t+3)', 'var9(t+3)', 'var10(t+3)',
                                                    'var11(t+3)', 'var12(t+3)', 'var1(t+4)', 'var2(t+4)',
                                                    'var4(t+4)', 'var5(t+4)', 'var6(t+4)', 'var7(t+4)',
                                                    'var8(t+4)', 'var9(t+4)', 'var10(t+4)', 'var11(t+4)', 'var12(t+4)',
                                                    'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var6(t+5)','var7(t+5)','var8(t+5)',
                                                    'var9(t+5)','var10(t+5)','var11(t+5)','var12(t+5)','var1(t+6)','var2(t+6)',
                                                    'var4(t+6)','var5(t+6)','var6(t+6)','var7(t+6)','var8(t+6)','var9(t+6)','var10(t+6)',
                                                    'var11(t+6)','var12(t+6)', 'var1(t+7)',
                                                    'var2(t+7)','var4(t+7)','var5(t+7)','var6(t+7)','var7(t+7)','var8(t+7)','var9(t+7)','var10(t+7)',
                                                    'var11(t+7)','var12(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                                    'var6(t+8)','var7(t+8)','var8(t+8)','var9(t+8)','var10(t+8)','var11(t+8)','var12(t+8)',
                                                    'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)','var6(t+9)','var7(t+9)','var8(t+9)','var9(t+9)',
                                                    'var10(t+9)','var11(t+9)','var12(t+9)'])
      elif nout == 10 and futuret == False and full == True:
            series_data = series_data.drop(columns=['var1(t+1)',
                                                    'var2(t+1)', 'var4(t+1)', 'var5(t+1)', 'var6(t+1)',
                                                    'var7(t+1)', 'var8(t+1)', 'var9(t+1)', 'var10(t+1)', 'var11(t+1)',
                                                    'var12(t+1)','var1(t+2)', 'var2(t+2)',
                                                    'var4(t+2)', 'var5(t+2)', 'var6(t+2)', 'var7(t+2)', 'var8(t+2)',
                                                    'var9(t+2)', 'var10(t+2)', 'var11(t+2)', 'var12(t+2)',
                                                    'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                                    'var6(t+3)', 'var7(t+3)', 'var8(t+3)', 'var9(t+3)', 'var10(t+3)',
                                                    'var11(t+3)', 'var12(t+3)', 'var1(t+4)', 'var2(t+4)',
                                                    'var4(t+4)', 'var5(t+4)', 'var6(t+4)', 'var7(t+4)',
                                                    'var8(t+4)', 'var9(t+4)', 'var10(t+4)', 'var11(t+4)', 'var12(t+4)',
                                                    'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var6(t+5)','var7(t+5)','var8(t+5)',
                                                    'var9(t+5)','var10(t+5)','var11(t+5)','var12(t+5)','var1(t+6)','var2(t+6)',
                                                    'var4(t+6)','var5(t+6)','var6(t+6)','var7(t+6)','var8(t+6)','var9(t+6)','var10(t+6)',
                                                    'var11(t+6)','var12(t+6)', 'var1(t+7)',
                                                    'var2(t+7)','var4(t+7)','var5(t+7)','var6(t+7)','var7(t+7)','var8(t+7)','var9(t+7)','var10(t+7)',
                                                    'var11(t+7)','var12(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                                    'var6(t+8)','var7(t+8)','var8(t+8)','var9(t+8)','var10(t+8)','var11(t+8)','var12(t+8)',
                                                    'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)','var6(t+9)','var7(t+9)','var8(t+9)','var9(t+9)',
                                                    'var10(t+9)','var11(t+9)','var12(t+9)', 'var13(t+1)','var13(t+2)','var13(t+3)','var13(t+4)',
                                                    'var13(t+5)','var13(t+6)','var13(t+7)','var13(t+8)','var13(t+9)'])
      elif nout==10 and futuret==True and full == False:
            series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)', 'var4(t+1)', 'var5(t+1)', 'var6(t+1)','var1(t+2)', 'var2(t+2)',
                                              'var4(t+2)', 'var5(t+2)', 'var6(t+2)',
                                              'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                              'var6(t+3)', 'var1(t+4)', 'var2(t+4)',
                                              'var4(t+4)', 'var5(t+4)', 'var6(t+4)',
                                              'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var6(t+5)','var1(t+6)','var2(t+6)',
                                              'var4(t+6)','var5(t+6)','var6(t+6)', 'var1(t+7)',
                                              'var2(t+7)','var4(t+7)','var5(t+7)','var6(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                              'var6(t+8)',
                                              'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)','var6(t+9)'])
      
      elif nout==10 and futuret==False and full == False:
            series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)', 'var4(t+1)', 'var5(t+1)','var1(t+2)', 'var2(t+2)',
                                              'var4(t+2)', 'var5(t+2)',
                                              'var1(t+3)', 'var2(t+3)', 'var4(t+3)', 'var5(t+3)',
                                              'var1(t+4)', 'var2(t+4)',
                                              'var4(t+4)', 'var5(t+4)',
                                              'var1(t+5)','var2(t+5)','var4(t+5)','var5(t+5)','var1(t+6)','var2(t+6)',
                                              'var4(t+6)','var5(t+6)','var1(t+7)',
                                              'var2(t+7)','var4(t+7)','var5(t+7)','var1(t+8)','var2(t+8)','var4(t+8)','var5(t+8)',
                                              'var1(t+9)','var2(t+9)','var4(t+9)','var5(t+9)',])
            
      elif nout==10 and futuret==False and empty == True:
            series_data = series_data.drop(columns=['var1(t+1)','var2(t+1)','var1(t+2)', 'var2(t+2)',
                                              'var1(t+3)', 'var2(t+3)', 
                                              'var1(t+4)', 'var2(t+4)',
                                              'var1(t+5)','var2(t+5)','var1(t+6)','var2(t+6)',
                                              'var1(t+7)',
                                              'var2(t+7)','var1(t+8)','var2(t+8)',
                                              'var1(t+9)','var2(t+9)',])
      else:
            print('Error! Check your shitty code')
            
      return series_data


