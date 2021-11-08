import numpy as np
import tensorflow as tf
from tensorflow import keras
from d2d import *
from datetime import datetime
import pickle as pkl
import json

def main():

    n = 10000  # The number of training examples to include in one fold.
    
    # Load the model definitions and datasets
    linear, lstm_model, dnn_model, df_all_chan,input_columns  = import_data()
    names = ('Multistep_Linear','Multistep_DNN','Multistep_LSTM')
    models = (linear,dnn_model,lstm_model)
    
    '''
    Run the analysis
    '''
    val_performance, performance, history, history_dict = k_fold(n,names,models,
                                                                 df_all_chan,input_columns)
    
    '''
    Save the loss curves into a figure
    '''
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    
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
    
    
    
    



if __name__ == "__main__":
    main()
