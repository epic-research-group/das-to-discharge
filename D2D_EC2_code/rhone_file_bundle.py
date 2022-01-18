from zipfile import ZipFile
import os
from os.path import basename

# create a ZipFile object
with ZipFile('rms_data_all_nofilt.zip', 'w') as zipObj:

   # Iterate over all the files in directory
   for folderName, subfolders, filenames in os.walk('rhone_processing'):

        for filename in filenames:
            #create complete filepath of file in directory
            filePath = os.path.join(folderName, filename)

            if filePath.endswith('.pkl'):
                # Add file to zip
                zipObj.write(filePath, basename(filePath))

            else:
                continue
