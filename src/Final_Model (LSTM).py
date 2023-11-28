# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:30:28 2023

@author: coope
"""

# Import Libraries
import numpy as np
import pandas as pd
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.data.Dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

from prophet import Prophet

import matplotlib
import matplotlib.pyplot as plt




# Load in data
energy = pd.read_csv('OneDrive\Desktop\IAA Material\Fall\Time Series II\Data\hrl_load_metered.csv', index_col=[0], parse_dates=[0])
energy_t1 = pd.read_csv('OneDrive\Desktop\IAA Material\Fall\Time Series II\Data\hrl_load_metered - test1.csv', index_col=[0], parse_dates=[0])
energy_t2 = pd.read_csv('OneDrive\Desktop\IAA Material\Fall\Time Series II\Data\hrl_load_metered - test2.csv', index_col=[0], parse_dates=[0])
energy_t3 = pd.read_csv('OneDrive\Desktop\IAA Material\Fall\Time Series II\Data\hrl_load_metered - test3.csv', index_col=[1], parse_dates=[1])
energy_t4 = pd.read_csv('OneDrive\Desktop\IAA Material\Fall\Time Series II\Data\hrl_load_metered - test4.csv', index_col=[0], parse_dates=[0])

energy_t3 = energy_t3.drop(['datetime_beginning_utc'], axis = 1)

# Combine dataframes for cleaning dataset
total_energy = pd.concat([energy, energy_t1, energy_t2, energy_t3, energy_t4])

# Drop unnessesary columns
aep_df = total_energy.drop(columns = ["nerc_region", "zone", "mkt_region", "load_area", "is_verified"])

# Rename Index
aep_df.index.names = ["Datetime"]

# Sort Data
aep_df.sort_index(inplace = True)

# Identify Duplicate Indices
duplicate_index = aep_df[aep_df.index.duplicated()]
print(aep_df.loc[duplicate_index.index.values, :])

# Replace Duplicates with Mean Value
aep_df = aep_df.groupby('Datetime').agg(np.mean)

#Set Datetime Index Frequency
aep_df = aep_df.asfreq('H')

# Determine # of Missing Values
print('# of Missing df_MW Values:{}'.format(len(aep_df[aep_df['mw'].isna()])))

# Impute Missing Values
aep_df['mw'] = aep_df['mw'].interpolate(limit_area = 'inside', limit = None)

# Create Train and Test Datasets
train = aep_df.loc[(aep_df.index >= datetime(2016, 1, 1)) & (aep_df.index < datetime(2023, 10, 12)), 'mw']
test = aep_df.loc[(aep_df.index >= datetime(2023, 10, 12)), 'mw']



# Long Short-Term Memory (LSTM) Neural Networks

np.random.seed(12345)



# Apply MinMaxScaler to training and test data. Fit to training data, use to transform train + test

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_train_values = scaler.fit_transform(train.values.reshape(-1, 1))
train_series_scaled = pd.Series(data = scaled_train_values.reshape(1, -1)[0], index = train.index)

# Create TimeseriesGenerator using training data and selected hyperparameters

n_input = 24
n_features = 1
sampling_rate = 1
stride = 1
batch_size = 1

train_generator = TimeseriesGenerator(scaled_train_values, 
                                      scaled_train_values, 
                                      length = n_input,
                                      sampling_rate = sampling_rate,
                                      stride = stride,
                                      batch_size = batch_size)

# Build and fit model
model = Sequential()
model.add(LSTM(100, activation = 'relu', return_sequences = False, input_shape = (n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

model.fit(train_generator, epochs = 80)





# Generate predictions

lstm_preds = []

batch = scaled_train_values[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    pred = model.predict(current_batch)[0]
    lstm_preds.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis = 1)
    
# Evaluate performance
    
lstm_preds = scaler.inverse_transform(lstm_preds)
lstm_preds = [item for sublist in lstm_preds for item in sublist]
    
test_score = mean_squared_error(test.values, lstm_preds)
print(test_score)




fig, axes = plt.subplots(figsize = (16,16))

LSTM_preds_series = pd.Series(lstm_preds, index = test.index)

axes.plot(test, label = 'Observed Values')
axes.plot(LSTM_preds_series, color = 'blue', label = 'Forecasted Values')
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Energy Consumption (MW)')
plt.legend()
axes.title.set_text('LSTM Forecasts')




lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])







