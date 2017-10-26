#!/usr/local/bin/python3

def read_data(data_filename):
    f = open(data_filename, 'rb')
    rows = int.from_bytes(f.read(4), 'little', signed=True)
    cols = int.from_bytes(f.read(4), 'little', signed=True)
    matrix = []
    for x in range(rows):
        row = []
        for y in range(cols):
            row.append(int.from_bytes(f.read(4), 'little', signed=True))
        matrix.append(row)
    return matrix
