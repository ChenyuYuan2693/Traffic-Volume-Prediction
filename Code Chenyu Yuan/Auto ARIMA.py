#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:54:36 2020

@author: chenyuyuan
"""


import pandas as pd
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
#print (unitroot_adf(traffic_flow_total.vehicle))
#plot_acf(traffic_flow_total)
#traffic_flow_total.plot(figsize = (200, 50))

train_1 = traffic_flow_total[0:2160]
test_1 = traffic_flow_total[2160:2880]
train_2 = traffic_flow_total[2880:5088]
test_2 = traffic_flow_total[5088:5832]
train_3 = traffic_flow_total[5832:8016]
test_3 = traffic_flow_total[8016:8592]
#%%
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import numpy as np
adf_test = ADFTest(alpha = 0.05)
print (adf_test.should_diff(train_1))


#%% 
def arimaPrediction(timeseries, prediction_num, test):
    timeseries_log = np.log(timeseries)
    arima_model = auto_arima(timeseries_log, start_p = 0, d = 1, start_q = 0, max_p = 5,
                         max_d = 5, max_q = 5, start_P = 0, D = 1, start_Q = 0,
                         max_P = 5, max_D = 5, max_Q = 5, m = 12, sensonal = True,
                         supress_warnings = True, stepwise = True, random_state = 20, n_fits = 50)
    prediction_1_log = pd.DataFrame(arima_model.predict(n_periods = prediction_num), index = test.index)
    prediction_1 = np.e**(prediction_1_log)
    prediction_1.columns = ['vehicle']
    return prediction_1, arima_model
def plot(true, prediction):
    plt.plot(prediction, color='red', label='predict_seq')
    plt.plot(true, color='blue', label='purchase_seq_test')
    plt.legend(loc='best')
    plt.show()
def rmse(true, prediction):
    true = np.asarray(true)
    prediction = np.asarray(prediction)
    rmse = np.sqrt(((prediction-true)**2).sum()/len(true))
    return rmse
#%%
prediction_1, model_1 = arimaPrediction(train_1, len(test_1), test_1)
prediction_2, model_2 = arimaPrediction(train_2, len(test_2), test_2)
prediction_3, model_3 = arimaPrediction(train_3, len(test_3), test_3)
#%%
plot(test_1['vehicle'], prediction_1['vehicle'])
plt.clf()
#%%
plot(test_2['vehicle'], prediction_2['vehicle'])
plt.clf()
#%%
plot(test_3['vehicle'], prediction_3['vehicle'])
   
#%%
rmse_1 = rmse(test_1, prediction_1)
rmse_2 = rmse(test_2, prediction_2)
rmse_3 = rmse(test_3, prediction_3)
    
#%%
print (model_1.summary())   
print (model_2.summary())
print (model_3.summary())
#%%
