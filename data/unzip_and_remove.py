import os
import zipfile
import multiprocessing


def myfunc(folder):
    os.chdir(folder)
    if os.path.exists('unzipped_and_removed'):
        os.chdir('..')
        return
    # an assumption is made that there is only one ZIP file per folder
    zip_file_path = [x for x in os.listdir() if x[-3:] == 'ZIP'][0]
    if os.path.exists(zip_file_path):
        zip_file_handle = zipfile.ZipFile(zip_file_path, 'r')
        zip_file_handle.extractall()
        os.remove(zip_file_path)
        # and remove the xml
        xml_file_path = [x for x in os.listdir() if x[-3:] == 'xml'][0]
        os.remove(xml_file_path)
        with open('unzipped_and_removed', 'w') as f:
            f.write('')
    os.chdir('..')


with open('folders_to_process.txt', 'r') as f:
    directories = f.read().splitlines()
os.chdir('preprocessing')
with multiprocessing.Pool(20) as p:
    p.map(myfunc, directories)
