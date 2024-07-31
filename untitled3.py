import pandas as pd
from functions import *
from model_functions import *
from the_model import *

from sklearn.metrics import mean_squared_error

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#F4511E", "#FB8C00", "#FFB300", "#FDD835", "#C0CA33", "#7CB342", "#43A047", "#00897B", "#00ACC1","#3949AB"])


series_data_vermont =  data_load_new('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Vermont/Vermont_Data_Combined.csv', 4, norm_ = True)

data_training_vermont = pd.DataFrame(data=series_data_vermont.iloc[0:600])
data_test_vermont = pd.DataFrame(data=series_data_vermont.iloc[600:674])


[vermont_data1,vermont_data2,vermont_labels1,vermont_labels2,vermont_labels10,vermont_labels20] = data_maker2(data_training_vermont, data_test_vermont)

[predictions_vermont_, vermont_error, vermont_hist1, vermont_hist2, model_vermont] = basic_prediction_model(vermont_data1, vermont_labels1, vermont_data2, vermont_labels20, prediction_type=1, batch_= 128) 

plt.figure(dpi = 600)
plt.plot(predictions_vermont_)
plt.plot(vermont_labels2, color='black')
plt.show()


plot_history(vermont_hist1)
plot_history(vermont_hist2)


test = model_vermont(vermont_data2, training = False)

test2 = model_vermont(hawaii_data1, training = False)

test3 = model_vermont(nevada_data1, training = False)

plt.figure(dpi = 600)
plt.plot(test)
plt.plot(vermont_labels2, color='black')
plt.show()

plt.figure(dpi = 600)
plt.plot(test2)
plt.plot(hawaii_labels1, color='black')
plt.show()

plt.figure(dpi = 600)
plt.plot(test3)
plt.plot(nevada_labels1, color='black')
plt.show()

mean_squared_error(vermont_labels2, test)

plot_error(test, vermont_labels2)

vermont_error


series_data_hawaii =  data_load_new('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Hawaii/Combined_Hawaii_Data.csv', 4, norm_ = True)

data_training_hawaii = pd.DataFrame(data=series_data_hawaii.iloc[0:600])
data_test_hawaii = pd.DataFrame(data=series_data_hawaii.iloc[600:674])


[hawaii_data1,hawaii_data2,hawaii_labels1,hawaii_labels2,hawaii_labels10,hawaii_labels20] = data_maker2(data_training_hawaii, data_test_hawaii)

[predictions_hawaii_, hawaii_error, hawaii_hist1, hawaii_hist2, model_hawaii] = basic_prediction_model(hawaii_data1, hawaii_labels1, hawaii_data2, hawaii_labels20, prediction_type=1, batch_= 128) 

plt.figure(dpi = 600)
plt.plot(predictions_hawaii_)
plt.plot(hawaii_labels2, color='black')
plt.show()


plot_history(hawaii_hist1)
plot_history(hawaii_hist2)

test_0 = model_hawaii(hawaii_data2, training = False)

test2_0 = model_hawaii(vermont_data1, training = False)

test3_0 = model_hawaii(nevada_data1, training = False)

hawaii_error

series_data_nevada =  data_load_new('C:/Users/Alfred/OneDrive - Noorda College of Osteopathic Medicine/Flu-modeling/Nevada/Nevada_Data_Combined.csv', 4, norm_ = True)

data_training_nevada = pd.DataFrame(data=series_data_nevada.iloc[0:400])
data_test_nevada = pd.DataFrame(data=series_data_nevada.iloc[400:500])


[nevada_data1,nevada_data2,nevada_labels1,nevada_labels2,nevada_labels10,nevada_labels20] = data_maker2(data_training_nevada, data_test_nevada)

[predictions_nevada_, nevada_error, nevada_hist1, nevada_hist2, model_nevada] = basic_prediction_model(nevada_data1, nevada_labels1, nevada_data2, nevada_labels20, prediction_type=1, batch_= 128) 

plt.figure(dpi = 600)
plt.plot(predictions_nevada_)
plt.plot(nevada_labels2, color='black')
plt.show()


plot_history(nevada_hist1)
plot_history(nevada_hist2)

nevada_error

test_1 = model_nevada(nevada_data2, training = False)

test2_1 = model_nevada(vermont_data1, training = False)

test3_1 = model_nevada(hawaii_data1, training = False)