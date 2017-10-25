from multiprocessing.pool import Pool

import laspy
import numpy
import os
from subsample_matrix import subsample_matrix


def fill_in_zeros(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            if matrix[i, j] == 0:
                # fill it in with the average of neighboring values
                total = 0
                number = 0
                if i > 0 and j > 0 and matrix[i-1, j-1] != 0:
                    total += matrix[i-1, j-1]
                    number += 1
                if j > 0 and matrix[i, j-1] != 0:
                    total += matrix[i, j-1]
                    number += 1
                if i < rows and j > 0 and matrix[i+1, j-1] != 0:
                    total += matrix[i+1, j-1]
                    number += 1

                if i > 0 and matrix[i-1, j] != 0:
                    total += matrix[i-1, j]
                    number += 1
                if i < rows and matrix[i+1, j] != 0:
                    total += matrix[i+1, j]
                    number += 1

                if i > 0 and j < cols and matrix[i-1, j+1] != 0:
                    total += matrix[i-1, j+1]
                    number += 1
                if j < cols and matrix[i, j+1] != 0:
                    total += matrix[i, j+1]
                    number += 1
                if i < rows and j < cols and matrix[i+1, j+1] != 0:
                    total += matrix[i+1, j+1]
                    number += 1

                if number > 0:
                    matrix[i, j] = int(total/number)


def count_zeros(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    counter = 0
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            if matrix[i, j] == 0:
                counter += 1
    print("Found", counter, "zeros")
    return counter


def convert(folder_name):
    os.chdir(folder_name)
    if os.path.exists('failed.txt'):
        try:
            las_filename = [x for x in os.listdir() if x[-3:] == 'las'][0]
            os.remove(las_filename)
        except:
            pass
        os.chdir('..')
        return

    las_filename = [x for x in os.listdir() if x[-3:] == 'las'][0]
    print(las_filename)
    f = laspy.file.File(las_filename)
    # so now we have f.X, f.Y, and f.Z, which should all be the same size. We need to make a matrix
    # whose size will be the square root of the length of one of these lists. To populate the matrix,
    # we go through the x,y,z lists and do the following: for the x-value, normalize it to the x-scale, and then
    # multiply it by the matrix size in order to get the matrix coordinate. Do the same for the y-value. Then place the
    # z-value inside the matrix. Finally, we'll save the matrix as a pickle or something like that
    # Note: it's important to use f.X and not f.x, since the former contains integers whereas the latter contains floats
    # obviously, computing with integers is much faster than with floats
    assert (len(f.Z) == len(f.X))
    assert (len(f.X) == len(f.Y))
    matrix_size = int(numpy.sqrt(len(f.Z)))  # int will also floor
    m = numpy.zeros([matrix_size, matrix_size])
    # grab the min/max X and Y values so as to do the calculation only once
    xmin = min(f.X)
    dx = max(f.X) - xmin
    ymin = min(f.Y)
    dy = max(f.Y) - ymin
    for i in range(0, len(f.Z)):
        x = f.X[i]
        col = int(numpy.floor((x - xmin) / dx * (matrix_size - 1)))
        y = f.Y[i]
        row = int(numpy.floor((y - ymin) / dy * (matrix_size - 1)))

        m[row, col] = f.Z[i]
        if i % 1000000 == 0:
            print(las_filename, i)
    if count_zeros(m) > 0:
        print(las_filename, "Filling in 0's...")
        fill_in_zeros(m)
    matrix_size = 500
    m = subsample_matrix(m, matrix_size)
    if count_zeros(m) > 0:
        print(las_filename, "Filling in 0's on subsampled matrix...")
        fill_in_zeros(m)
    # Not going to worry too much about filling in 0's. They're annoying, but there should be significantly more
    # meaningful data than 0 data, and hopefully the net figures out to ignore the 0 data
    m.dump(las_filename.split('.')[0] + '.pickle')
    os.remove(las_filename)
    with open('pickled', 'w') as f:
        f.write('')  # This file indicates success to the pipeline
    os.chdir('..')


with open('folders_to_process.txt', 'r') as f:
    directories = f.read().splitlines()
os.chdir('preprocessing')
with Pool(20) as p:
    p.map(convert, directories)
