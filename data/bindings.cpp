#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "convert_las_to_matrix.hpp"
#include "interpolate_zeros.hpp"

namespace py = pybind11;


PYBIND11_MODULE(utils, module)
{
  module.def("convert_las_to_matrix_and_store", &converter::convert_las_to_matrix_and_store, "converts a given LAS file to a matrix and saves output to disk");
  module.def("interpolate_zeros", &matrix_ops::interpolate_zeros, "takes in a matrix and fills all the 0-valued elements with interpolations from nearby values");
}
