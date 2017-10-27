#include "pybind11/pybind11.h"
#include "convert_las_to_matrix.hpp"
#include "read_data.hpp"

namespace py = pybind11;


PYBIND11_MODULE(utils, module)
{
  module.def("convert_las_to_matrix_and_store", &converter::convert_las_to_matrix_and_store, "converts a given LAS file to a matrix and saves output to disk");
  module.def("read_data", &utils::read_data, "read rows and columns from the provided .data file, returns numpy array");
}
