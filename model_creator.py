import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from d2d import *
from compile_fit import *


filename = "/data/fast0/datasets/Rhone_data_continuous.h5"
f = h5py.File(filename, 'r')
print("Keys: %s" % f.keys())

das_data_all = f['DAS Data'][:]
discharge = f['Discharge'][:]

df_all_chan = pd.DataFrame(das_data_all)
df_all_chan['Discharge'] = discharge

column_indices = {name: i for i, name in enumerate(df_all_chan.columns)}

input_columns = list(np.arange(0,2308,1))

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

dnn_model = tf.keras.models.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
])