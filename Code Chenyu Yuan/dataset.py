#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:33:30 2020

@author: chenyuyuan
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


#%%

# import dataset
import pandas as pd
file = pd.read_csv('/Users/chenyuyuan/Introduction to Python Prototyping/Traffic Station/dot_traffic_2015.txt')
file_2 = pd.read_csv('/Users/chenyuyuan/Introduction to Python Prototyping/Traffic Station/dot_traffic_stations_2015.txt')
#%%

# data overview
print (file.shape) #(7140391, 38)
print (file.columns)
'''
Index(['date', 'day_of_data', 'day_of_week', 'direction_of_travel',
       'direction_of_travel_name', 'fips_state_code',
       'functional_classification', 'functional_classification_name',
       'lane_of_travel', 'month_of_data', 'record_type', 'restrictions',
       'station_id', 'traffic_volume_counted_after_0000_to_0100',
       'traffic_volume_counted_after_0100_to_0200',
       'traffic_volume_counted_after_0200_to_0300',
       'traffic_volume_counted_after_0300_to_0400',
       'traffic_volume_counted_after_0400_to_0500',
       'traffic_volume_counted_after_0500_to_0600',
       'traffic_volume_counted_after_0600_to_0700',
       'traffic_volume_counted_after_0700_to_0800',
       'traffic_volume_counted_after_0800_to_0900',
       'traffic_volume_counted_after_0900_to_1000',
       'traffic_volume_counted_after_1000_to_1100',
       'traffic_volume_counted_after_1100_to_1200',
       'traffic_volume_counted_after_1200_to_1300',
       'traffic_volume_counted_after_1300_to_1400',
       'traffic_volume_counted_after_1400_to_1500',
       'traffic_volume_counted_after_1500_to_1600',
       'traffic_volume_counted_after_1600_to_1700',
       'traffic_volume_counted_after_1700_to_1800',
       'traffic_volume_counted_after_1800_to_1900',
       'traffic_volume_counted_after_1900_to_2000',
       'traffic_volume_counted_after_2000_to_2100',
       'traffic_volume_counted_after_2100_to_2200',
       'traffic_volume_counted_after_2200_to_2300',
       'traffic_volume_counted_after_2300_to_2400', 'year_of_data'],
      dtype='object')
'''
summary = file.describe()

#%%
for i in range(13, 37):
    column_name = file.columns[i]
    if (i == 13):
        data = file[file[column_name] <= 19800]
        
    else:
        data = data[data[column_name] <= 19800]
    data = data[data[column_name] >= 0]
#%%
summary_after_filter = data.describe()
road_describe = data['functional_classification_name'].unique()

#%%
PA_data = data[data['fips_state_code'] == 42]
urban_interstate = PA_data[PA_data['functional_classification_name'] == 'Urban: Principal Arterial - Interstate']
rural_interstate = PA_data[PA_data['functional_classification_name'] == 'Rural: Principal Arterial - Interstate']
urban_minor = PA_data[PA_data['functional_classification_name'] == 'Urban: Minor Arterial']
rural_minor = PA_data[PA_data['functional_classification_name'] == 'Rural: Minor Arterial']
#%%
#hourly data time series
def hourTimeSeries(PA_data):
    PA_data_timeseries = PA_data.iloc[:,13:37]
    time = list(range(0, 24))
    traffic_ave = PA_data_timeseries.mean(axis = 0).tolist()
    traffic_ave_df = pd.DataFrame({'time':time, 'traffic_ave':traffic_ave})
    return traffic_ave_df


#%%
#weekly data time series
def weeklyTimeSeires(PA_data):
    time = list(range(1, 8))
    PA_weekly = PA_data.groupby(by = ['day_of_week'])
    PA_weekly_ave = PA_weekly.mean().iloc[:,7:31]
    PA_weekly_ave = PA_weekly_ave.sum(axis = 1)
    PA_weekly_ave_df = pd.DataFrame({'time':time, 'traffic_ave':PA_weekly_ave})
    return PA_weekly_ave_df

#%%
#weekly data time series
def dailyTimeSeires(PA_data):
    time = list(range(1, 32))
    PA_daily = PA_data.groupby(by = ['day_of_data'])
    PA_daily_ave = PA_daily.mean().iloc[:,7:31]
    PA_daily_ave = PA_daily_ave.sum(axis = 1)
    PA_daily_ave_df = pd.DataFrame({'time':time, 'traffic_ave':PA_daily_ave})
    return PA_daily_ave_df


#%%
#monthly data time series
def monthlyTimeSeries(PA_data):
    time = list(range(1, 13))
    PA_monthly = PA_data.groupby(by = ['month_of_data', 'day_of_data'])
    PA_monthly_ave = PA_monthly.mean().iloc[:,7:31]
    PA_monthly_ave = PA_monthly_ave.sum(axis = 1)
    PA_monthly_ave = PA_monthly_ave.groupby(by = ['month_of_data'])
    PA_monthly_ave = PA_monthly_ave.sum()
    PA_monthly_ave_df = pd.DataFrame({'time':time, 'traffic_ave':PA_monthly_ave})
    return PA_monthly_ave_df



#%%
def plot(traffic_ave_df, PA_daily_ave_df, PA_weekly_ave_df, PA_monthly_ave_df):
    plt.figure(figsize = (40, 40))
    
    
    # #plot monthly time series line
    plt.subplot(411)
    plt.plot(traffic_ave_df['time'], traffic_ave_df['traffic_ave'])
    plt.xlim((0,23))
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(True)
    plt.xlabel('hours (h)', fontsize = 30)
    plt.ylabel('traffic volume (count)', fontsize = 30)
    plt.title('Average traffic volume in an hourly scale in Pennsylvania', fontsize = 30)
    
    #plot daily time series line
    plt.subplot(412)
    plt.plot(PA_daily_ave_df['time'], PA_daily_ave_df['traffic_ave'])
    plt.xlim((1,31))
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(True)
    plt.xlabel('daily (d)', fontsize = 30)
    plt.ylabel('average traffic volume (count)', fontsize = 30)
    plt.title('Average traffic volume in an daily scale in Pennsylvania', fontsize = 30)
    
    #plot weekly time series line
    plt.subplot(413)
    plt.plot(PA_weekly_ave_df['time'], PA_weekly_ave_df['traffic_ave'])
    plt.xlim((1,7))
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(True)
    plt.xlabel('week', fontsize = 30)
    plt.ylabel('average traffic volume (count)', fontsize = 30)
    plt.title('Average traffic volume in an weekly scale in Pennsylvania', fontsize = 30)
    
    #plot monthly time series line
    plt.subplot(414)
    plt.plot(PA_monthly_ave_df['time'], PA_monthly_ave_df['traffic_ave'])
    plt.xlim((1,12))
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(True)
    plt.xlabel('month (mo)', fontsize = 30)
    plt.ylabel('average traffic volume (count)', fontsize = 30)
    plt.title('Average traffic volume in an monthly scale in Pennsylvania', fontsize = 30)
    
    plt.savefig('/Users/chenyuyuan/Introduction to Python Prototyping/5.jpg')
    

    
#%%
traffic_ave_df_urban_interstate = hourTimeSeries(urban_interstate)
PA_daily_ave_df_urban_interstate = dailyTimeSeires(urban_interstate)
PA_weekly_ave_df_urban_interstate = weeklyTimeSeires(urban_interstate)
PA_monthly_ave_df_urban_interstate = monthlyTimeSeries(urban_interstate)
plot(traffic_ave_df_urban_interstate, PA_daily_ave_df_urban_interstate, PA_weekly_ave_df_urban_interstate, PA_monthly_ave_df_urban_interstate)
#%%
traffic_ave_df_rural_interstate = hourTimeSeries(rural_interstate)
PA_daily_ave_df_rural_interstate = dailyTimeSeires(rural_interstate)
PA_weekly_ave_df_rural_interstate = weeklyTimeSeires(rural_interstate)
PA_monthly_ave_df_rural_interstate = monthlyTimeSeries(rural_interstate)
plot(traffic_ave_df_rural_interstate, PA_daily_ave_df_rural_interstate, PA_weekly_ave_df_rural_interstate, PA_monthly_ave_df_rural_interstate)
#%%
traffic_ave_df_urban_minor = hourTimeSeries(urban_minor)
PA_daily_ave_df_urban_minor = dailyTimeSeires(urban_minor)
PA_weekly_ave_df_urban_minor = weeklyTimeSeires(urban_minor)
PA_monthly_ave_df_urban_minor = monthlyTimeSeries(urban_minor)
plot(traffic_ave_df_urban_minor, PA_daily_ave_df_urban_minor, PA_weekly_ave_df_urban_minor, PA_monthly_ave_df_urban_minor)
#%%       
traffic_ave_df_rural_minor = hourTimeSeries(rural_minor)
PA_daily_ave_df_rural_minor = dailyTimeSeires(rural_minor)
PA_weekly_ave_df_rural_minor = weeklyTimeSeires(rural_minor)
PA_monthly_ave_df_rural_minor = monthlyTimeSeries(rural_minor)
plot(traffic_ave_df_rural_minor, PA_daily_ave_df_rural_minor, PA_weekly_ave_df_rural_minor, PA_monthly_ave_df_rural_minor)
