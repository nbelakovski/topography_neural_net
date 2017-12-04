#pragma once

#include <pybind11/numpy.h>

namespace matrix_ops
{
/***
 * @brief Given a matrix, fill any 0-values with surrounding data (interpolated as necessary, i.e. 1 0 0 4 becomes 1 2 3 4)
 * @return 0 on success. -1 if there is a row filled entirely with 0's
 **/
int interpolate_zeros(pybind11::array_t<int> & matrix);
void interpolate_zeros_2(pybind11::array_t<int> & matrix);
}

