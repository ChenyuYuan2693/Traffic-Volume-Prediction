import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import math
#%%
file = pd.read_csv('/Users/chenyuyuan/Introduction to Python Prototyping/Traffic Station/dot_traffic_2015.txt')
file_2 = pd.read_csv('/Users/chenyuyuan/Introduction to Python Prototyping/Traffic Station/dot_traffic_stations_2015.txt')
#%%
for i in range(13, 37):
    column_name = file.columns[i]
    if (i == 13):
        data = file[file[column_name] <= 19800]
        
    else:
        data = data[data[column_name] <= 19800]
    data = data[data[column_name] >= 0]
    
#%%
data_by_station = data.groupby(by = ['station_id', 'fips_state_code'])
count = data_by_station.size()
big_data_count = count[count >= 500]
big_data_count = pd.DataFrame(big_data_count)
big_data_count = big_data_count.sort_values(0, ascending = False)
#%%
import datetime
import statsmodels.api as sm
index = ('000C7L', 15)
station = data_by_station.get_group(index)
station['date'] = pd.to_datetime(station['date'], format = '%Y-%m-%d')
station.sort_values('date', inplace = True)
station_north = station[station['direction_of_travel_name'] == 'East']
station_south = station[station['direction_of_travel_name'] == 'West']
station_north_groupby = station_north.groupby(by = ['date'])
station_south_groupby = station_south.groupby(by = ['date'])
mean_station_north = station_north_groupby.mean()
mean_station_south = station_south_groupby.mean()
#%%
i = 0
for index, row in mean_station_north.iterrows():
    date = index
    traffic_flow = row[8:32]
    traffic_flow = pd.DataFrame(traffic_flow)
    traffic_flow.columns = [0]
    if (i == 0):
        traffic_flow_total = pd.DataFrame(traffic_flow)
    else:
        traffic_flow_total = traffic_flow_total.append(traffic_flow)
    i += 1
    
#traffic_flow_total = pd.Index(sm.tsa.datetools.dates_from_range('2015-01-01'))
#print (sm.tsa.datetools.dates_from_range('2015-01-01 00:00:00', '2015-12-31 23:00:00'))


#%%
import datetime


def dateHourRange(beginDateHour, endDateHour):
    dhours = []
    dhour = datetime.datetime.strptime(beginDateHour, "%Y-%m-%d %H")
    date = beginDateHour[:]
    while date <= endDateHour:
        dhours.append(date)
        dhour = dhour + datetime.timedelta(hours=1)
        date = dhour.strftime("%Y-%m-%d %H")
    return dhours
data_range = dateHourRange("2015-01-01 00", "2015-12-31 23")

traffic_flow_total.index = data_range[0:len(traffic_flow_total)]
traffic_flow_total.columns = ['vehicle']

#%%
train_1 = traffic_flow_total[0:2160]
test_1 = traffic_flow_total[2160:2880]
train_2 = traffic_flow_total[2880:5088]
test_2 = traffic_flow_total[5088:5832]
train_3 = traffic_flow_total[5832:8016]
test_3 = traffic_flow_total[8016:8592]
#%%
train_1_np = np.asarray(train_1)
train_2_np = np.asarray(train_2[0:2160])
train_3_np = np.asarray(train_3[0:2160])
#%%
test_1_np = np.asarray(test_1)
test_2_np = np.asarray(test_2)
test_3_np = np.asarray(test_3)


#%%
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):

        end_element_index = i + n_steps_in
        out_end_index = end_element_index + n_steps_out
        
        if out_end_index > len(sequences): 
            break
        
        sequence_x, sequence_y = sequences[i:end_element_index,:], sequences[end_element_index:out_end_index,:]
        X.append(sequence_x)
        y.append(sequence_y)

    return np.array(X), np.array(y)

def multi_parallel_output_model(n_input, n_output, X, y, epochs_num):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs_num, verbose=0)
    return model

#%%
def prediction (train, test):
    steps_in, steps_out = len(test), len(test)
    X_time, y_time = split_sequences(train, steps_in, steps_out)
    n_input = X_time.shape[1]*X_time.shape[2]
    X_time = X_time.reshape((X_time.shape[0], n_input))
    n_output = y_time.shape[1] * y_time.shape[2]
    y_time = y_time.reshape((y_time.shape[0], n_output))
    model = multi_parallel_output_model(n_input, n_output, X_time, y_time, 200) 
    x_input = test
    x_input = x_input.reshape((1, n_input))
    yhat = model.predict(x_input, verbose = 0)
    yhat = yhat.transpose()
    return yhat
def plot(yhat, ytrue):
    plt.plot(yhat[0:720], color='red', label='predict_seq')
    plt.plot(ytrue[0:720], color='blue', label='purchase_seq_test')
    plt.legend(loc='best')
    plt.show()
def rmse(yhat, ytrue):
    result = math.sqrt(((yhat-ytrue)**2).sum() / len(yhat))
    return result

#%%
yhat_1 = prediction(train_1_np, test_1_np)
yhat_2 = prediction(train_2_np, test_2_np)
yhat_3 = prediction(train_3_np, test_3_np)


#%%
plot(yhat_1, test_1_np)
plot(yhat_2, test_2_np)
plot(yhat_3, test_3_np)

#%%
rmse_1 = rmse(yhat_1, test_1_np)
rmse_2 = rmse(yhat_2, test_2_np)
rmse_3 = rmse(yhat_3, test_3_np)


