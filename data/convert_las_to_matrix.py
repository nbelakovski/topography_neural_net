from multiprocessing.pool import Pool
import sys
import numpy
import os
from utils import convert_las_to_matrix_and_store

# I dislike modifying the path like this, but it's the most straightforward way to get this module loaded
# without a more serious rearrangment of the directory structure or adding some dependcy like a proper PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), '..', 'tools'))
from tools import read_data


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
        # open up the file and count the 0's in the matrix
        # If there are too many, reject it
        m = read_data(out_filename)
        zeros = m.size - numpy.count_nonzero(m)
        if zeros < 125000:
            with open('pickled', 'w') as f:
                f.write('')  # This file indicates success to the pipeline
        else:
            with open('failed.txt', 'a') as f:
                f.write("Found " + str(zeros) + " zeros")


with open(sys.argv[1], 'r') as f:
    directories = f.read().splitlines()
with Pool(3) as p:
    p.map(convert, directories)
