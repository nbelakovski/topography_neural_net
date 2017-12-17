from multiprocessing.pool import Pool
import sys
import numpy
import os
from utils import convert_las_to_matrix_and_store


def convert(folder_name):
    os.chdir(folder_name)
    las_filename = ''
    try:
        las_filename = [x for x in os.listdir() if x[-3:] == 'las'][0]
    except:
        return

    if os.path.exists('failed.txt') or not os.path.exists('cropped_size.txt'):  # This probably indicated the crop failed. In this case, remove the las file so it doesn't take up space
        os.remove(las_filename)
        return

    out_filename = las_filename.split('.')[0] + '.data'
    with open('cropped_size.txt','r') as f:
         [desired_rows, desired_cols, channels] = [int(x) for x in f.readline().split(',')]
    success = convert_las_to_matrix_and_store(las_filename, desired_rows, desired_cols, out_filename)
    os.remove(las_filename)
    if success:
        with open('pickled', 'w') as f:
            f.write('')  # This file indicates success to the pipeline


with open(sys.argv[1], 'r') as f:
    directories = f.read().splitlines()
with Pool(3) as p:
    p.map(convert, directories)
