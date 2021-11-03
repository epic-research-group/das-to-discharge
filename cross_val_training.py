import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from d2d import *
from model_creator import *
from compile_fit import *
from datetime import datetime
import pickle as pkl
import json

# datetime object containing current date and time

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

#Plotting
p = 1

fig = plt.figure(figsize=(12, 28), facecolor='w', edgecolor='k')

#chunking and defining the data

n = 10000  #chunk row size
list_df = [df_all_chan[i:i+n] for i in range(0,df_all_chan.shape[0],n)]

val_performance={}
performance={}
history={}
history_dict = {}

#cross validation training

k = 1
a = 1


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
    
    #Linear model
    
    history['linear_fold'+str(k)] = compile_and_fit(linear, multi_step_window)

    val_performance['Multistep_Linear_fold' + str(k)] = linear.evaluate(multi_step_window.val)
    performance['Multistep_Linear_fold' + str(k)] = linear.evaluate(multi_step_window.test, verbose=0)
    
    history_dict['Multistep_Linear_fold' + str(k) + '_loss'] = history['linear_fold'+str(k)].history['loss']
    history_dict['Multistep_Linear_fold' + str(k) + '_val_loss'] = history['linear_fold'+str(k)].history['val_loss']
    
    plt.subplot(7, 3, a)
    plt.plot(history['linear_fold'+str(k)].history['loss'], label='loss')
    plt.plot(history['linear_fold'+str(k)].history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.title("Loss Curve Linear, fold: "+str(k))
    a += 1
    
    #DNN model
    
    history['dnn_fold'+str(k)] = compile_and_fit(dnn_model, multi_step_window)

    val_performance['Multistep_DNN_fold' + str(k)] = dnn_model.evaluate(multi_step_window.val)
    performance['Multistep_DNN_fold' + str(k)] = dnn_model.evaluate(multi_step_window.test, verbose=0)
    
    history_dict['DNN_fold' + str(k) + '_loss'] = history['dnn_fold'+str(k)].history['loss']
    history_dict['DNN_fold' + str(k) + '_val_loss'] = history['dnn_fold'+str(k)].history['val_loss']
    
    plt.subplot(7, 3, a)
    plt.plot(history['dnn_fold'+str(k)].history['loss'], label='loss')
    plt.plot(history['dnn_fold'+str(k)].history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.title("Loss Curve DNN, fold: "+str(k))
    a += 1
                
    #LSTM model
    
    history['lstm_fold'+str(k)] = compile_and_fit(lstm_model, multi_step_window)

    val_performance['Multistep_LSTM_fold' + str(k)] = lstm_model.evaluate(multi_step_window.val)
    performance['Multistep_LSTM_fold' + str(k)] = lstm_model.evaluate(multi_step_window.test, verbose=0)
    
    history_dict['LSTM_fold' + str(k) + '_loss'] = history['lstm_fold'+str(k)].history['loss']
    history_dict['LSTM_fold' + str(k) + '_val_loss'] = history['lstm_fold'+str(k)].history['val_loss']
    
    plt.subplot(7, 3, a)
    plt.plot(history['lstm_fold'+str(k)].history['loss'], label='loss')
    plt.plot(history['lstm_fold'+str(k)].history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.title("Loss Curve DNN, fold: "+str(k))
    a += 1
    
    print('Done with fold: ' + str(k))
    
    k += 1

    
#saving the loss curves into a figure

plt.tight_layout()
plt.savefig("loss_curves/" + "loss_curves_"+dt_string+".png", dpi=300, bbox_inches='tight')

#saving the performance metrics

file = open('performance_metrics/performance_metrics'+dt_string+'.txt', 'w')
file.write('performance: ' + str(performance) + '    ')
file.write('val_performance: ' + str(val_performance))
file.close()

#saving histories, losses into a pickle file

with open('history_losses/history_losses_'+dt_string+'.pkl', 'wb') as hist_f:
    pkl.dump(history_dict, hist_f)

#saving the models

linear.save('saved_models/linear_model_h5_'+dt_string+'.h5')
dnn_model.save('saved_models/dnn_model_h5_'+dt_string+'.h5')
lstm_model.save('saved_models/lstm_model_h5_'+dt_string+'.h5')

print('Done! Wrote metrics to performance_metrics.txt and saved models in /saved_models')