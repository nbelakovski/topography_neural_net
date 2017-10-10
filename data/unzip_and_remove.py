import os
import zipfile

# noinspection PyArgumentList
data_directories = [x for x in os.listdir() if x.isdigit()]
data_directories.sort(key=lambda x: int(x))
for directory in data_directories:
    # an assumption is made that there is only one ZIP file per folder
    zip_file_path = directory + '/' + [x for x in os.listdir(directory) if x[-3:] == 'ZIP'][0]
    if os.path.exists(zip_file_path):
        zip_file_handle = zipfile.ZipFile(zip_file_path, 'r')
        zip_file_handle.extractall(directory)
        os.remove(zip_file_path)
    if int(directory) % 25 == 0:
        print("Processed", int(directory), "files")
