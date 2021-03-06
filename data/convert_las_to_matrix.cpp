// Always include the corresponding header as the first substantive line of code in the source file
#include "convert_las_to_matrix.hpp"
// Project headers
// Third party headers
#include <liblas/liblas.hpp>
// C/C++ STL headers
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

using namespace std;

namespace converter
{

bool convert_las_to_matrix_and_store(const std::string & las_filename, const uint32_t desired_rows, const uint32_t desired_cols, const std::string & out_filename)
{
    int32_t ** matrix_ptr = convert_las_to_matrix(las_filename, desired_rows, desired_cols);
    if (matrix_ptr)
    {
        write_matrix_to_file(out_filename, matrix_ptr, desired_rows, desired_cols);
        // Now deallocate the memory
        for (uint32_t i = 0; i < desired_rows; ++i)
        {
            delete[] matrix_ptr[i];
        }
        delete[] matrix_ptr;
        return true;
    }
    else
    {
        return false;
    }
}

int32_t ** convert_las_to_matrix(const string & las_filename, const uint32_t desired_rows, const uint32_t desired_cols)
{

    // Load the LAS file
    ifstream las_file;
    las_file.open(las_filename.c_str(), ios::in | ios::binary);
    liblas::ReaderFactory f;
    liblas::Reader r = f.CreateWithStream(las_file);

    // Get the total number of points contained in the file
    const liblas::Header h = r.GetHeader();
    const uint32_t total_points = h.GetPointRecordsCount();

    // Allocate the matrix based on the specs
    if (total_points< (desired_rows * desired_cols))
    {
        cerr << "Not enough points for desired matrix size. Trash" << endl;
        return NULL;
    }
    int32_t ** matrix = new int32_t*[desired_rows];
    for (uint32_t i = 0; i < desired_rows; ++i)
    {
        matrix[i] = new int32_t[desired_cols];
        memset(matrix[i], 0, sizeof(int32_t) * desired_cols);
    }

    // populate the matrix
    // first get relevant info
    const double xscale = h.GetScaleX();
    const double xoffset = h.GetOffsetX();
    const double yscale = h.GetScaleY();
    const double yoffset = h.GetOffsetY();
    
    // sticking with int's since they're faster
    // usage of scale/offset coming from documentation: https://www.liblas.org/python.html?highlight=scale#liblas.header.Header.scale
    int xmin = (h.GetMinX() - xoffset) / xscale;
    int dx = ((h.GetMaxX() - xoffset) / xscale) - xmin;
    int ymin = (h.GetMinY() - yoffset) / yscale;
    int dy = ((h.GetMaxY() - yoffset) / yscale) - ymin;

    while (r.ReadNextPoint())
    {
        liblas::Point p = r.GetPoint();
        int x = p.GetRawX();
        int col = float(x - xmin) / dx * (desired_cols - 1);
        int y = p.GetRawY();
        int row = float(y - ymin) / dy * (desired_rows - 1);
        matrix[row][col] = p.GetRawZ();
    }

    return matrix;
}

void write_matrix_to_file(const string & out_filename, const int32_t * const * matrix, const uint32_t rows, const uint32_t cols)
{
    ofstream out_file;
    out_file.open(out_filename.c_str(), ios::out | ios::binary);
    // Write the matrix size to the first byte
    out_file.write(reinterpret_cast<const char *>(&rows), sizeof(int));
    out_file.write(reinterpret_cast<const char *>(&cols), sizeof(int));
    for (uint32_t i = 0; i < rows; ++i)
    {
        out_file.write(reinterpret_cast<const char *>(matrix[i]), cols * sizeof(int32_t));
    }
}
} // close namespace converter
