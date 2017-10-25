#!/usr/local/bin/python3

def read_data(data_filename):
	f = open(data_filename, 'rb')
	matrix_size = int.from_bytes(f.read(4), 'little', signed=True)
	matrix = []
	for x in range(matrix_size):
		row = []
		for y in range(matrix_size):
			row.append(int.from_bytes(f.read(4), 'little', signed=True))
		matrix.append(row)
	return matrix
