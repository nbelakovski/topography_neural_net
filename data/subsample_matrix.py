import numpy


# Return a square matrix of shape (matrix_size, matrix_size) populated with values from original_matrix.
# Not intended to work with upsampling
def subsample_matrix(original_matrix, matrix_size):
    m = numpy.zeros([matrix_size, matrix_size])
    for row in range(0, matrix_size):
        zi = int(row/matrix_size * (original_matrix.shape[0] - 1))
        for col in range(0, matrix_size):
            zj = int(col/matrix_size * (original_matrix.shape[1] - 1))
            m[row, col] = original_matrix[zi, zj]
    return m
