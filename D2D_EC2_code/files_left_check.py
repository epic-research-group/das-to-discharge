
import boto3
import os


s3 = boto3.resource('s3')
bucket = s3.Bucket('rhoneglacier-eth')
bucket_name = 'rhoneglacier-eth'


list_of_files = []

for s3_objects in bucket.objects.all():
    fn_string = s3_objects.key
    list_of_files.append(fn_string)


completed_files = os.listdir()

reformed_completed_files = []

new_strings = []

for element in completed_files:
    
    new_string = element.replace('_rms_nofilt.pkl','')
    new_strings.append(new_string)

for i in new_strings:
    rest = 'new/' + i[10:18]+'/'+i
    reformed_completed_files.append(rest)

remaining_files = list(set(list_of_files) - set(reformed_completed_files))

print(len(remaining_files))

print(remaining_files)
