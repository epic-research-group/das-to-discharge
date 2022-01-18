import boto3
import os
from scipy import signal
import h5py
import numpy as np
import pickle
from time import perf_counter
from pathos.multiprocessing import ProcessPool


s3 = boto3.resource('s3')
bucket = s3.Bucket('rhoneglacier-eth')
bucket_name = 'rhoneglacier-eth'

sample_rate = 1e3
channel_spacing = 4



t1_start = perf_counter()

list_of_files = []

for s3_objects in bucket.objects.all():
    fn_string = s3_objects.key
    list_of_files.append(fn_string)

#for bucket_object in bucket.objects.all():


completed_files = os.listdir('.')

reformed_completed_files = []

new_strings = []

for element in completed_files:
    
    new_string = element.replace('_rms_nofilt.pkl','')
    new_strings.append(new_string)

for i in new_strings:
    rest = 'new/' + i[10:18]+'/'+i
    reformed_completed_files.append(rest)





remaining_files = list(set(list_of_files) - set(reformed_completed_files))




input_vector = remaining_files

def rms_on_s3(file_path):

    #file_path = bucket_object.key
    filename = os.path.basename(file_path)


    if filename.endswith('.hdf5'):

        s3.meta.client.download_file(bucket_name, file_path, filename)

        array = []

        with h5py.File(filename, 'r') as data:

            data_retrieved = 116. / 8192. * sample_rate / 10. * np.transpose(np.array(data['/raw_das_data']))
            #t = np.arange(data_retrieved.shape[1]) / sample_rate
            #x = np.arange(data_retrieved.shape[0]) * channel_spacing

            data_detrended = signal.detrend(data_retrieved,axis=1)

            #sos = signal.butter(2, 10, 'highpass', fs = 1000, output='sos')
            #filtered = signal.sosfilt(sos, data_detrended)
            rms = np.sqrt(np.mean((data_detrended)**2,axis=1))

            array.append(rms)
    else:
        return

    with open(str(filename) + '_rms_nofilt.pkl', 'wb') as file:

        pickle.dump(array, file)

    print ('wrote '+ str(filename) + '_rms_nofilt.pkl to file!')

    os.remove(filename)

    #testing_count += 1
p = ProcessPool(nodes=1)

p.map(rms_on_s3, input_vector)


t1_stop = perf_counter()

print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
