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
from model_creator import *

def plot_loss(history, model = 'linear'):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves/" + model + "_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.clf()

n = 10000  #chunk row size
list_df = [df_all_chan[i:i+n] for i in range(0,df_all_chan.shape[0],n)]

val_performance={}
performance={}
history = {}

k = 1

for i in list_df:
    n = len(i)
    labels = list(i.index)
    
    df_all_chan_copy = df_all_chan.copy()
    
    train_df = df_all_chan_copy.drop(labels=labels, axis=0)
    
    val_df = i[int(n*0.0):int(n*0.6)]
    test_df = i[int(n*0.6):int(n*1.0)]
    
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    multi_step_window = WindowGenerator(
        input_width=200, label_width=1, shift=0,
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df,
        label_columns=['Discharge'],
        input_columns=input_columns)
    
    history['linear_fold'+str(k)] = compile_and_fit(linear, multi_step_window)

    val_performance['Multistep_Linear_fold' + str(k)] = linear.evaluate(multi_step_window.val)
    performance['Multistep_Linear_fold' + str(k)] = linear.evaluate(multi_step_window.test, verbose=0)
    
    plot_loss(history['linear_fold'+str(k)], model = 'linear_fold'+str(k))
    
    history['dnn_fold'+str(k)] = compile_and_fit(dnn_model, multi_step_window)

    val_performance['Multistep_DNN_fold' + str(k)] = dnn_model.evaluate(multi_step_window.val)
    performance['Multistep_DNN_fold' + str(k)] = dnn_model.evaluate(multi_step_window.test, verbose=0)
    
    plot_loss(history['dnn_fold'+str(k)], model = 'dnn_fold' + str(k))
    
    history['lstm_fold'+str(k)] = compile_and_fit(lstm_model, multi_step_window)

    val_performance['Multistep_LSTM_fold' + str(k)] = lstm_model.evaluate(multi_step_window.val)
    performance['Multistep_LSTM_fold' + str(k)] = lstm_model.evaluate(multi_step_window.test, verbose=0)
    
    plot_loss(history['lstm_fold'+str(k)], model = 'lstm_fold' + str(k))
    
    print('Done with fold: ' + str(k))
    
    k += 1
    
    


file = open('performance_metrics.txt', 'a')
file.write('performance: ' + str(performance) + '    ')
file.write('val_performance: ' + str(val_performance))
file.close()

linear.save('saved_models/linear_model_h5.h5')
dnn_model.save('saved_models/dnn_model_h5.h5')
lstm_model.save('saved_models/lstm_model_h5.h5')

print('Done! Wrote metrics to performance_metrics.txt and saved models in /saved_models')