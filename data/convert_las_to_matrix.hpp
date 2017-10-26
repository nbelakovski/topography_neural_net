#pragma once

#include <stdint.h>
#include <string>

namespace converter
{
bool convert_las_to_matrix_and_store(const std::string & las_filename, const uint32_t desired_rows, const uint32_t desired_cols, const std::string & out_filename);
int32_t ** convert_las_to_matrix(const std::string & las_filename, const uint32_t desired_rows, const uint32_t desired_cols);
void write_matrix_to_file(const std::string & out_filename, const int32_t * const * matrix, const uint32_t rows, const uint32_t cols);
} // close namespace converter
