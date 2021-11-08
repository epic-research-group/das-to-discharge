import numpy as np
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.keras import layers

class WindowGenerator():
    
    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)

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


    def __init__(self, input_width, label_width, shift,
               train_df, 
               val_df, 
               test_df,
               label_columns=None,
               input_columns=None):
        
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        
        # Do the same for the input column indices.
        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in
                                        enumerate(input_columns)}
        self.input_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

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

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Input column name(s): {self.input_columns}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

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
    
def k_fold(n,names,models,data,input_columns,early_stop=np.nan):    
    '''
    Run a k-fold analysis on folds of size n
    '''
    
    list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]

    val_performance={}
    performance={}
    history={}
    history_dict = {}

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

        val_df = this_data[int(n*0.0):int(n*0.6)]
        test_df = this_data[int(n*0.6):int(n*1.0)]

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

        '''
        Loop over the model types
        '''
        
        for this_name,this_model in zip(names,models):

            history[name+str(k)] = compile_and_fit(this_model, multi_step_window)
            val_performance[name +'_fold' + str(k)] = linear.evaluate(multi_step_window.val)
            performance[name + '_fold' + str(k)] = linear.evaluate(multi_step_window.test, 
                                                                            verbose=0)
            history_dict[name + '_fold' + str(k) + '_loss'] = \
                history[name + '_fold'+str(k)].history['loss']

            history_dict['Multistep_Linear_fold' + str(k) + '_val_loss'] = \
                history[name + '_fold'+str(k)].history['val_loss']


        print('Done with fold: ' + str(k))

    return val_performance, performance, history, history_dict

def compile_and_fit(model, window, patience=2):
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

def import_data(filename = "/data/fast0/datasets/Rhone_data_continuous.h5"):
    
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
    
    return linear, lstm_model, dnn_model, df_all_chan, input_columns