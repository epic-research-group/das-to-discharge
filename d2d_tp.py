import numpy as np
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.keras import layers
from tensorflow.math import reduce_prod
import matplotlib

class WindowGenerator():
    
#     @property
#     def train(self):
#       return self.make_dataset(self.train_df)

#     @property
#     def val(self):
#       return self.make_dataset(self.val_df)

#     @property
#     def test(self):
#       return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)

        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


    def __init__(self, df, input_width, label_width, shift,
                   label_columns=None, input_columns_DAS=None, input_columns_tp=None,
                    shuffle=True, batch_size = 16):
        
        # Store the raw data.
#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df



        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(df.columns)}
        
        # Do the same for the input column indices for DAS.
        self.input_columns_DAS = input_columns_DAS
        if input_columns_DAS is not None:
            self.input_columns_indices_DAS = {name: i for i, name in
                                        enumerate(input_columns_DAS)}
        self.input_indices_DAS = {name: i for i, name in
                               enumerate(df.columns)}
        
        # Do the same for the input column indices for Temp and Precip.
        self.input_columns_tp = input_columns_tp
        if input_columns_tp is not None:
            self.input_columns_indices_tp = {name: i for i, name in
                                        enumerate(input_columns_tp)}
        self.input_indices_tp = {name: i for i, name in
                               enumerate(df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        ds_DAS, ds_tp = self.make_dataset(df,shuffle=shuffle, batch_size=batch_size)
        
#         print(ds_DAS)
#         print(ds_tp)
        
        # Split the dataset
        train_split=0.7
        val_split=0.2
        test_split=0.1
        
        ds_size = len(ds_DAS)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        test_size = int(test_split * ds_size)
        
        train_ds = ds_DAS.take(train_size)
        val_ds = ds_DAS.skip(train_size).take(val_size)
        test_ds = ds_DAS.skip(train_size).skip(val_size)
        
        train_ds_tp = ds_tp.take(train_size)
        val_ds_tp = ds_tp.skip(train_size).take(val_size)
        test_ds_tp = ds_tp.skip(train_size).skip(val_size)
        
        self.train_ds_tp = train_ds_tp
        self.val_ds_tp = val_ds_tp
        self.test_ds_tp = test_ds_tp
        
        #Redoing the normalization for DAS
        
        train_strain_in_one = []
        train_dis_in_one = []
        
        for i in train_ds.as_numpy_iterator():
            train_strain_in_one.append(i[0])
            train_dis_in_one.append(i[1])

        
        train_strain_in_one = np.asarray(train_strain_in_one)
        train_dis_in_one = np.asarray(train_dis_in_one)
        
        #print(train_dis_in_one[0])
        
        train_strain_in_one = np.reshape(train_strain_in_one, (train_strain_in_one.shape[0]*train_strain_in_one.shape[1] * input_width, 2308))
        train_dis_in_one = np.reshape(train_dis_in_one, (train_dis_in_one.shape[0]*train_dis_in_one.shape[1] * label_width, label_width, 1))
        
        chan_mean = np.mean(train_strain_in_one, axis = 0)
        dis_mean = np.mean(train_dis_in_one)
        
        chan_std = np.std(train_strain_in_one, axis = 0)
        dis_std = np.std(train_dis_in_one)
                 
        train_channels_normed = []
        train_discharge_normed = []
        
        for element in train_ds.as_numpy_iterator():
            norm_chan = (element[0] - chan_mean) / chan_std
            norm_dis = (element[1] - dis_mean) / dis_std
            train_channels_normed.append(norm_chan)
            train_discharge_normed.append(norm_dis)
        
        val_channels_normed = []
        val_discharge_normed = []        
        
        for element in val_ds.as_numpy_iterator():
            norm_chan = (element[0] - chan_mean) / chan_std
            norm_dis = (element[1] - dis_mean) / dis_std
            val_channels_normed.append(norm_chan)
            val_discharge_normed.append(norm_dis)        

        test_channels_normed = []
        test_discharge_normed = []        
        
        for element in test_ds.as_numpy_iterator():
            norm_chan = (element[0] - chan_mean) / chan_std
            norm_dis = (element[1] - dis_mean) / dis_std
            test_channels_normed.append(norm_chan)
            test_discharge_normed.append(norm_dis)
            
        
        
        #Do the normalization for Temp and Precip data
        train_tp_in_one = []
        train_dis_in_one_tp = []
        
        for i in train_ds_tp.as_numpy_iterator():
            train_tp_in_one.append(i[0])
            train_dis_in_one_tp.append(i[1])
            
        train_tp_in_one = np.asarray(train_tp_in_one)
        train_dis_in_one_tp = np.asarray(train_dis_in_one_tp)
        
        #print(train_dis_in_one_tp[0])
        
        train_tp_in_one = np.reshape(train_tp_in_one, (train_tp_in_one.shape[0]*train_tp_in_one.shape[1] * input_width, 2))
        

        train_dis_in_one_tp = np.reshape(train_dis_in_one_tp, (train_dis_in_one_tp.shape[0]*train_dis_in_one_tp.shape[1] * label_width, label_width, 1))

        
        tp_mean = np.mean(train_tp_in_one, axis = 0)
#         print(tp_mean)
        dis_mean_tp = np.mean(train_dis_in_one_tp)
#         print(dis_mean_tp)
        
        tp_std = np.std(train_tp_in_one, axis = 0)
        dis_std_tp = np.std(train_dis_in_one_tp)
        
        self.tp_mean = tp_mean
        self.dis_mean_tp = dis_mean_tp
        
        self.tp_std = tp_std
        self.dis_std_tp = dis_std_tp
        
        
        train_tp_normed = []
        train_tp_dis_normed = []
        
        for element in train_ds_tp.as_numpy_iterator():
            norm_tp = (element[0] - tp_mean) / tp_std
            norm_dis = (element[1] - dis_mean_tp) / dis_std_tp
            train_tp_normed.append(norm_tp)
            train_tp_dis_normed.append(norm_dis)
            
        
        val_tp_normed = []
        val_tp_dis_normed = []        
        
        for element in val_ds_tp.as_numpy_iterator():
            norm_tp = (element[0] - tp_mean) / tp_std
            norm_dis = (element[1] - dis_mean_tp) / dis_std_tp
            val_tp_normed.append(norm_tp)
            val_tp_dis_normed.append(norm_dis)        

        test_tp_normed = []
        test_tp_dis_normed = []        
        
        for element in test_ds_tp.as_numpy_iterator():
            norm_tp = (element[0] - tp_mean) / tp_std
            norm_dis = (element[1] - dis_mean_tp) / dis_std_tp
            test_tp_normed.append(norm_tp)
            test_tp_dis_normed.append(norm_dis)
        
            
        #Check if the last array is oddly shaped and will not fit into the model
        if np.asarray(test_tp_normed)[-1].shape != np.asarray(test_tp_normed)[0].shape:

            test_tp_normed.pop()
            test_tp_dis_normed.pop()  
            
        if np.asarray(test_discharge_normed)[-1].shape != np.asarray(test_discharge_normed)[0].shape:

            test_channels_normed.pop()
            test_discharge_normed.pop()  
        
        train_dataset_normed = tf.data.Dataset.from_tensor_slices((train_channels_normed, train_discharge_normed))
        val_dataset_normed = tf.data.Dataset.from_tensor_slices((val_channels_normed, val_discharge_normed))
        test_dataset_normed = tf.data.Dataset.from_tensor_slices((test_channels_normed, test_discharge_normed))
        
        train_dataset_normed_tp = tf.data.Dataset.from_tensor_slices((train_tp_normed, train_tp_dis_normed))
        val_dataset_normed_tp = tf.data.Dataset.from_tensor_slices((val_tp_normed, val_tp_dis_normed))
        test_dataset_normed_tp = tf.data.Dataset.from_tensor_slices((test_tp_normed, test_tp_dis_normed))
            
#         print(std_chan_mean)
#         print(train_chan_mean)
#         print(train_dis_mean)
#         print(std_dis_mean)
        
    
        self.train_ds_tp = train_ds_tp
        self.val_ds_tp = val_ds_tp
        self.test_ds_tp = test_ds_tp
    
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
    
        self.ds_DAS = ds_DAS
        self.training_non_normed = train_ds
        self.train_strain_in_one = train_strain_in_one
        self.train_channels_normed = train_channels_normed
        self.train_dis_in_one = train_dis_in_one
    
        self.train = train_dataset_normed
        self.val = val_dataset_normed
        self.test = test_dataset_normed
        
        self.train_tp = train_dataset_normed_tp
        self.val_tp = val_dataset_normed_tp
        self.test_tp = test_dataset_normed_tp
        
        self.chan_mean = chan_mean
        self.chan_std = chan_std
        self.dis_mean = dis_mean
        self.dis_std = dis_std
        

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Input column name(s): {self.input_columns}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    

        
    
    
    
    def make_dataset(self, data, shuffle, batch_size):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.input_width,
            shuffle=shuffle,
            seed = 1,
            batch_size = batch_size) #default is 32

        ds_DAS = ds.map(self.split_window_DAS)
        ds_tp = ds.map(self.split_window_tp)

        return ds_DAS, ds_tp
    
    def split_window_DAS(self, ds):
        inputs = ds[:, self.input_slice, :]
        labels = ds[:, self.labels_slice, :]
#         print(inputs)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        if self.input_columns_DAS is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns_DAS],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def split_window_tp(self, ds):
        inputs = ds[:, self.input_slice, :]
        labels = ds[:, self.labels_slice, :]
#         print(inputs)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        if self.input_columns_tp is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns_tp],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    

def compile_and_fit_das(model, window, patience=10, MAX_EPOCHS = 1000, learning_rate = 0.001):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])

    return history

def compile_and_fit_tp(model, window, patience=10, MAX_EPOCHS = 1000, learning_rate = 0.001):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train_tp, epochs=MAX_EPOCHS,
                      validation_data=window.val_tp,
                      callbacks=[early_stopping])

    return history  



# NOT IN USE

# def k_fold_leave_out(n,names,models,data,input_columns,early_stop=np.nan,window_input_width = 200, learning_rate = 0.001):    
#     '''
#     Run a k-fold analysis on folds of size n
#     '''
    
#     list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]

#     val_performance={}
#     performance={}
#     history={}
#     history_dict = {}
#     running_stats = pd.DataFrame()
        
# #     train_mean =  np.zeros( len(list_df) )
# #     train_std =  np.zeros( len(list_df) )
# #     test_mean =  np.zeros( len(list_df) )
# #     test_std =  np.zeros( len(list_df) )
# #     val_mean =  np.zeros( len(list_df) )
# #     val_std =  np.zeros( len(list_df) )

#     #cross validation training

#     '''
#     Loop over the folds
#     '''
#     for k,this_data in enumerate(list_df):
#         if not np.isnan(early_stop):
#             if early_stop == k:
#                 break
            
#         n = len(this_data)
#         labels = list(this_data.index)

#         data_copy = data.copy()

#         train_mean, train_std, test_mean, test_std, val_mean, val_std,\
#                 train_df, val_df, test_df = simple_split(this_data)

#         running_stats['Fold'+str(k)+'_train_mean'] = train_mean
#         running_stats['Fold'+str(k)+'_train_std'] = train_std
#         running_stats['Fold'+str(k)+'_val_mean'] = val_mean
#         running_stats['Fold'+str(k)+'_val_std'] = val_std
#         running_stats['Fold'+str(k)+'_test_mean'] = test_mean
#         running_stats['Fold'+str(k)+'_test_std'] = test_std

#         multi_step_window = WindowGenerator(
#             input_width=window_input_width, label_width=1, shift=0,
#             data=this_data, 
#             label_columns=['Discharge'],
#             input_columns=input_columns)

#         '''
#         Loop over the model types
#         '''
        
#         for this_name, this_model in zip(names,models):

#             history[this_name + str(k)] = compile_and_fit(this_model, multi_step_window, learning_rate = learning_rate)
#             val_performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.val)
#             performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.test, 
#                                                                             verbose=0)
#             history_dict[this_name + '_fold' + str(k)] = \
#                 history[this_name + str(k)].history['loss']

#             history_dict[this_name + '_fold' + str(k) + '_val_loss'] = \
#                 history[this_name + str(k)].history['val_loss']


        
        
        
# #         k_fold_stats = {'mean_train':train_mean,
# #                       'std_train':train_std,
# #                       'mean_val':val_mean,
# #                       'std_val':val_std,
# #                       'mean_test':test_mean,
# #                       'std_test':test_std} 
        
#         print('Done with fold: ' + str(k)+', chunk size: '+ str(n))

#     return val_performance, performance, history, history_dict, running_mean
# """

# """
# NOT IN USE

# def k_fold(n,names,models,data,input_columns,early_stop=np.nan,window_input_width = 200, learning_rate = 0.001):    
#     '''
#     Run a k-fold analysis on folds of size n
#     '''
    
#     list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]

#     val_performance={}
#     performance={}
#     history={}
#     history_dict = {}
    
# #     train_mean = np.zeros( len(list_df) )
# #     train_std = np.zeros( len(list_df) )
# #     test_mean = np.zeros( len(list_df) )
# #     test_std = np.zeros( len(list_df) )
# #     val_mean = np.zeros( len(list_df) )
# #     val_std = np.zeros( len(list_df) )
    

#     #cross validation training

#     '''
#     Loop over the folds
#     '''
#     for k,this_data in enumerate(list_df):
#         if not np.isnan(early_stop):
#             if early_stop == k:
#                 break
            
#         n = len(this_data)
#         labels = list(this_data.index)

#         data_copy = data.copy()

#         train_df = data_copy.drop(labels=labels, axis=0)
#         train_mean = train_df.mean()
#         train_std = train_df.std()
        
#         val_df = this_data[int(n*0.0):int(n*0.6)]
#         val_mean = val_df.mean()
#         val_std = val_df.std()
        
#         test_df = this_data[int(n*0.6):int(n*1.0)]
#         test_mean = test_df.mean()
#         test_std = test_df.std()


#         train_df = (train_df - train_mean) / train_std
#         val_df = (val_df - train_mean) / train_std
#         test_df = (test_df - train_mean) / train_std

#         multi_step_window = WindowGenerator(
#             input_width=window_input_width, label_width=1, shift=0,
#             train_df=train_df, 
#             val_df=val_df, 
#             test_df=test_df,
#             label_columns=['Discharge'],
#             input_columns=input_columns)

#         '''
#         Loop over the model types
#         '''
        
#         for this_name, this_model in zip(names,models):

#             history[this_name + str(k)] = compile_and_fit(this_model, multi_step_window, learning_rate = learning_rate)
#             val_performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.val)
#             performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.test, 
#                                                                             verbose=0)
#             history_dict[this_name + '_fold' + str(k) + '_loss'] = \
#                 history[this_name + str(k)].history['loss']

#             history_dict[this_name + '_fold' + str(k) + '_val_loss'] = \
#                 history[this_name + str(k)].history['val_loss']


#         print('Done with fold: ' + str(k))
        
# #     k_fold_stats = {'mean_train':train_mean,
# #                       'std_train':train_std,
# #                       'mean_val':val_mean,
# #                       'std_val':val_std,
# #                       'mean_test':test_mean,
# #                       'std_test':test_std}

#     return val_performance, performance, history, history_dict



def import_data(filename = "/data/fast0/datasets/Rhone_data_continuous.h5"):
    
    #Read in the Temp and Precip data
    temp = pd.read_pickle(r'temp_array.pkl')
    times_for_temp = pd.read_pickle(r'dates_4_temp.pkl')
    precip =  pd.read_pickle(r'precip_mm.pkl')
    
    #Read in the DAS data
    f = h5py.File(filename, 'r')
    
    #Get the discharge times
    times_of_discharge = matplotlib.dates.date2num(f['Times'][:])
    
    print("Keys: %s" % f.keys())
    
    #Assign variables to DAS and Discharge data
    das_data_all = f['DAS Data'][:]
    discharge = f['Discharge'][:]
    
    #Make a Pandas dataframe of the data
    df_all_chan = pd.DataFrame(das_data_all)
    df_all_chan['Discharge'] = discharge
    
    #Interpolate Temp and Precip to match extent of Discharge measurements
    temp_at_dis = np.interp(times_of_discharge, times_for_temp, temp)
    precip_at_dis = np.interp(times_of_discharge, times_for_temp, precip)
    
    #Add the interpolated precip and temp data to the data frame
    df_all_chan['Temperature'] = temp_at_dis
    df_all_chan['Precipitation'] = precip_at_dis

    column_indices = {name: i for i, name in enumerate(df_all_chan.columns)}

    input_columns_DAS = list(np.arange(0,2308,1))
    input_columns_tp = ['Temperature', 'Precipitation']
    
    linear_model_tp = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    
    linear_model_DAS = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    
    lstm_model_tp = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences = True),
        #tf.keras.layers.LSTM(32, return_sequences=True),
        #tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(1)
    ])

    lstm_model_DAS = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences = True),
        #tf.keras.layers.LSTM(32, return_sequences=True),
        #tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(1)
    ])

    dnn_model_DAS = tf.keras.Sequential([
         
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        tf.keras.layers.Flatten(),
        layers.Dense(1),
          
    ])
    
#     conv_model =  tf.keras.Sequential([
#         tf.keras.layers.Conv1D(filters=32,
#                            kernel_size=(200,), #(window_input _width integer, )
#                            activation='relu'),
#         tf.keras.layers.Dense(units=32, activation='relu'),
#         tf.keras.layers.Dense(units=1),
# ])
    
    return linear_model_DAS, linear_model_tp, lstm_model_DAS, lstm_model_tp, dnn_model_DAS, df_all_chan, input_columns_DAS, input_columns_tp, das_data_all, f