import numpy as np
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.keras import layers

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
                   label_columns=None, input_columns=None,
                    shuffle=False, batch_size = 16):
        
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
        
        # Do the same for the input column indices.
        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                        enumerate(input_columns)}
        self.input_indices = {name: i for i, name in
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
        
        ds = self.make_dataset(df,shuffle=shuffle, batch_size=batch_size)
        
        # Split the dataset
        train_split=0.7
        val_split=0.2
        test_split=0.1
        
        ds_size = len(ds)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        test_size = int(test_split * ds_size)
        
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        
        #Redoing the normalization
        
        train_strain_in_one = []
        train_dis_in_one = []
        
        for i in train_ds.as_numpy_iterator():
            train_strain_in_one.append(i[0])
            train_dis_in_one.append(i[1])
            
        train_strain_in_one = np.asarray(train_strain_in_one)
        train_dis_in_one = np.asarray(train_dis_in_one)
        
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
            
            
        
            
        # Doing the normalization

        
#         stds_of_window = []
#         sums_chan = []        
#         sums_dis = []
#         stds_of_dis = []
        
#         for element in train_ds.as_numpy_iterator():
            
#             sums_chan.append(element[0].sum(axis=1))
#             sums_dis.append(element[1].sum(axis=0))
            
#             stds_of_window.append(element[0].std(axis=1))
#             stds_of_dis.append(element[1].std(axis=0))
        
        
#         sums_of_chan_sums = []
#         sums_of_dis_sums = []
#         mean_stds_of_batches = []
#         mean_stds_of_dis = []
        
#         for element in sums_dis:
        
#             sums_of_dis_sums.append(element.sum(axis=0))
            
#         for element in stds_of_dis:
            
#             mean_stds_of_dis.append(element.mean(axis=0))
        
#         for element in stds_of_window:
            
#             mean_stds_of_batches.append(element.mean(axis=0))
            
#         for element in sums_chan:
            
#             sums_of_chan_sums.append(element.sum(axis=0))

        
#         total_chan_sum = np.asarray(sums_of_chan_sums).sum(axis=0)
#         total_dis_sum = np.asarray(sums_of_dis_sums).sum(axis=0)
        
#         std_chan_mean = np.asarray(mean_stds_of_batches).mean(axis=0)
#         std_dis_mean = np.asarray(mean_stds_of_dis).mean(axis=0)
#         train_chan_mean = total_chan_sum / (self.total_window_size * np.asarray(sums_chan).shape[1] * np.asarray(sums_of_chan_sums).shape[0])
#         train_dis_mean = total_dis_sum / (self.total_window_size*np.asarray(sums_dis).shape[1] * np.asarray(sums_of_dis_sums).shape[0])
        
#         print(train_chan_mean)
        
#         train_channels_normed = []
#         train_discharge_normed = []
        
#         for element in train_ds.as_numpy_iterator():
#             print(element[0].shape)
#             print(element[1].shape)
#             norm_chan = (element[0] - train_chan_mean) / std_chan_mean
#             norm_dis = (element[1] - train_dis_mean) / std_dis_mean
#             train_channels_normed.append(norm_chan)
#             train_discharge_normed.append(norm_dis)
        
#         val_channels_normed = []
#         val_discharge_normed = []        
        
#         for element in val_ds.as_numpy_iterator():
#             norm_chan = (element[0] - train_chan_mean) / std_chan_mean
#             norm_dis = (element[1] - train_dis_mean) / std_dis_mean
#             val_channels_normed.append(norm_chan)
#             val_discharge_normed.append(norm_dis)        

#         test_channels_normed = []
#         test_discharge_normed = []        
        
#         for element in test_ds.as_numpy_iterator():
#             norm_chan = (element[0] - train_chan_mean) / std_chan_mean
#             norm_dis = (element[1] - train_dis_mean) / std_dis_mean
#             test_channels_normed.append(norm_chan)
#             test_discharge_normed.append(norm_dis)        
        
        #Check if the last array is oddly shaped and will not fit into the model
        if np.asarray(test_discharge_normed)[-1].shape != np.asarray(test_discharge_normed)[0].shape:

            test_channels_normed.pop()
            test_discharge_normed.pop()  
        
        train_dataset_normed = tf.data.Dataset.from_tensor_slices((train_channels_normed, train_discharge_normed))
        val_dataset_normed = tf.data.Dataset.from_tensor_slices((val_channels_normed, val_discharge_normed))
        test_dataset_normed = tf.data.Dataset.from_tensor_slices((test_channels_normed, test_discharge_normed))
    
            
#         print(std_chan_mean)
#         print(train_chan_mean)
#         print(train_dis_mean)
#         print(std_dis_mean)
        self.train_strain_in_one = train_strain_in_one
        self.train_channels_normed = train_channels_normed
    
        self.train = train_dataset_normed
        self.val = val_dataset_normed
        self.test = test_dataset_normed
        
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

        ds = ds.map(self.split_window)

        return ds
    
    def split_window(self, ds):
        inputs = ds[:, self.input_slice, :]
        labels = ds[:, self.labels_slice, :]
#         print(inputs)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    

def compile_and_fit(model, window, patience=10, MAX_EPOCHS = 100, learning_rate = 0.001):

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


"""
NOT IN USE

def k_fold_leave_out(n,names,models,data,input_columns,early_stop=np.nan,window_input_width = 200, learning_rate = 0.001):    
    '''
    Run a k-fold analysis on folds of size n
    '''
    
    list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]

    val_performance={}
    performance={}
    history={}
    history_dict = {}
    running_stats = pd.DataFrame()
        
#     train_mean =  np.zeros( len(list_df) )
#     train_std =  np.zeros( len(list_df) )
#     test_mean =  np.zeros( len(list_df) )
#     test_std =  np.zeros( len(list_df) )
#     val_mean =  np.zeros( len(list_df) )
#     val_std =  np.zeros( len(list_df) )

    #cross validation training

    '''
    Loop over the folds
    '''
    for k,this_data in enumerate(list_df):
        if not np.isnan(early_stop):
            if early_stop == k:
                break
            
        n = len(this_data)
        labels = list(this_data.index)

        data_copy = data.copy()

        train_mean, train_std, test_mean, test_std, val_mean, val_std,\
                train_df, val_df, test_df = simple_split(this_data)

        running_stats['Fold'+str(k)+'_train_mean'] = train_mean
        running_stats['Fold'+str(k)+'_train_std'] = train_std
        running_stats['Fold'+str(k)+'_val_mean'] = val_mean
        running_stats['Fold'+str(k)+'_val_std'] = val_std
        running_stats['Fold'+str(k)+'_test_mean'] = test_mean
        running_stats['Fold'+str(k)+'_test_std'] = test_std

        multi_step_window = WindowGenerator(
            input_width=window_input_width, label_width=1, shift=0,
            data=this_data, 
            label_columns=['Discharge'],
            input_columns=input_columns)

        '''
        Loop over the model types
        '''
        
        for this_name, this_model in zip(names,models):

            history[this_name + str(k)] = compile_and_fit(this_model, multi_step_window, learning_rate = learning_rate)
            val_performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.val)
            performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.test, 
                                                                            verbose=0)
            history_dict[this_name + '_fold' + str(k)] = \
                history[this_name + str(k)].history['loss']

            history_dict[this_name + '_fold' + str(k) + '_val_loss'] = \
                history[this_name + str(k)].history['val_loss']


        
        
        
#         k_fold_stats = {'mean_train':train_mean,
#                       'std_train':train_std,
#                       'mean_val':val_mean,
#                       'std_val':val_std,
#                       'mean_test':test_mean,
#                       'std_test':test_std} 
        
        print('Done with fold: ' + str(k)+', chunk size: '+ str(n))

    return val_performance, performance, history, history_dict, running_mean
"""

"""
NOT IN USE

def k_fold(n,names,models,data,input_columns,early_stop=np.nan,window_input_width = 200, learning_rate = 0.001):    
    '''
    Run a k-fold analysis on folds of size n
    '''
    
    list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]

    val_performance={}
    performance={}
    history={}
    history_dict = {}
    
#     train_mean = np.zeros( len(list_df) )
#     train_std = np.zeros( len(list_df) )
#     test_mean = np.zeros( len(list_df) )
#     test_std = np.zeros( len(list_df) )
#     val_mean = np.zeros( len(list_df) )
#     val_std = np.zeros( len(list_df) )
    

    #cross validation training

    '''
    Loop over the folds
    '''
    for k,this_data in enumerate(list_df):
        if not np.isnan(early_stop):
            if early_stop == k:
                break
            
        n = len(this_data)
        labels = list(this_data.index)

        data_copy = data.copy()

        train_df = data_copy.drop(labels=labels, axis=0)
        train_mean = train_df.mean()
        train_std = train_df.std()
        
        val_df = this_data[int(n*0.0):int(n*0.6)]
        val_mean = val_df.mean()
        val_std = val_df.std()
        
        test_df = this_data[int(n*0.6):int(n*1.0)]
        test_mean = test_df.mean()
        test_std = test_df.std()


        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        multi_step_window = WindowGenerator(
            input_width=window_input_width, label_width=1, shift=0,
            train_df=train_df, 
            val_df=val_df, 
            test_df=test_df,
            label_columns=['Discharge'],
            input_columns=input_columns)

        '''
        Loop over the model types
        '''
        
        for this_name, this_model in zip(names,models):

            history[this_name + str(k)] = compile_and_fit(this_model, multi_step_window, learning_rate = learning_rate)
            val_performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.val)
            performance[this_name + '_fold' + str(k)] = this_model.evaluate(multi_step_window.test, 
                                                                            verbose=0)
            history_dict[this_name + '_fold' + str(k) + '_loss'] = \
                history[this_name + str(k)].history['loss']

            history_dict[this_name + '_fold' + str(k) + '_val_loss'] = \
                history[this_name + str(k)].history['val_loss']


        print('Done with fold: ' + str(k))
        
#     k_fold_stats = {'mean_train':train_mean,
#                       'std_train':train_std,
#                       'mean_val':val_mean,
#                       'std_val':val_std,
#                       'mean_test':test_mean,
#                       'std_test':test_std}

    return val_performance, performance, history, history_dict
"""


def import_data(filename = "/data/fast0/datasets/Rhone_data_continuous.h5"):
    
    f = h5py.File(filename, 'r')
    print("Keys: %s" % f.keys())

    das_data_all = f['DAS Data'][:]
    discharge = f['Discharge'][:]

    df_all_chan = pd.DataFrame(das_data_all)
    df_all_chan['Discharge'] = discharge

    column_indices = {name: i for i, name in enumerate(df_all_chan.columns)}

    input_columns = list(np.arange(0,2308,1))

    linear_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    dnn_model = tf.keras.models.Sequential([
         
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
    
    return linear_model, lstm_model, dnn_model, df_all_chan, input_columns