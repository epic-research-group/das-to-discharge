import numpy as np
import tensorflow as tf
import random
import sys
import pickle
import d2d
import importlib
importlib.reload(d2d)
import pandas as pd
import os

def main():

    history = {}
    val_performance = {}
    performance = {}

    seeds = list(np.arange(1,26,1))
    input_columns = [list(np.arange(0, 2308, 1)), 
                     list(np.arange(0, 1600, 1)), 
                     list(np.arange(1600, 2308, 1))]
    location = ['Whole Cable', 'Ablation Zone', 'Accumulation Zone']

    for SEED in seeds:

        for ic, loc in zip(input_columns, location):


            file = "/data/fast0/datasets/Rhone_data_continuous_highpass.h5"

            filt = 'Highpass'

            ww = 200
            bs = 32

            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)


            linear_model, lstm_model, \
            dnn_model, df_all_chan, \
            das_data_all, f  = d2d.import_data(filename = file, 
                                               input_columns = ic)

            da = df_all_chan.to_numpy()

            multi_step_window_shuffled = d2d.WindowGenerator(df_all_chan,
                                                            input_width=ww,
                                                            label_width=1, 
                                                            shift=0,
                                                            label_columns=['Discharge'],
                                                            input_columns=ic,
                                                            shuffle=True,
                                                            batch_size=bs)

            history['lstm '+loc+' '+str(SEED)] = d2d.compile_and_fit(lstm_model, 
                                                                     multi_step_window_shuffled, 
                                                                     learning_rate = 0.001)

            val_performance['lstm '+loc+' '+str(SEED)] =\
                            lstm_model.evaluate(multi_step_window_shuffled.val)
            performance['lstm '+loc+' '+str(SEED)] = lstm_model.evaluate(multi_step_window_shuffled.test,
                                                                         verbose=0)
    output = {'history':history,
              'val_performance':val_performance,
              'performance':performance}
    
    with open('glacier_zone_comparison.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
if __name__ == "__main__":
    main()
