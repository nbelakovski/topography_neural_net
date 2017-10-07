import laspy
import numpy
import os


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


def convert(folder_name):
    las_filename = folder_name + '/' + [x for x in os.listdir(folder_name) if x[-3:] == 'las'][0]
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
            print(i)
    print("Done, counting 0's")
    count_zeros(m)
    print("Filling in 0's...")
    fill_in_zeros(m)  # fill in 0's here since it's better to do it with full resolution data than after subsampling
    print("Filling in 0's...done")
    count_zeros(m)
    m.dump(las_filename.split('.')[0] + '.pickle')


# noinspection PyArgumentList
data_directories = [x for x in os.listdir() if x.isdigit()]
for directory in data_directories:
    convert(directory)
