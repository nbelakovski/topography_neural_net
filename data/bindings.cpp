#include "pybind11/pybind11.h"
#include "convert_las_to_matrix.hpp"

namespace py = pybind11;


PYBIND11_MODULE(utils, module)
{
	module.def("convert_las_to_matrix_and_store", &converter::convert_las_to_matrix_and_store, "converts a given LAS file to a matrix and saves output to disk");
}