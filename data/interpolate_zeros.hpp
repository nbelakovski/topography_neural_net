#pragma once

#include <pybind11/numpy.h>
/*
// forward declarations (the "hidden" attribute is to suppress compiler warnings, similar to how pybind11 does it in detail/common.h
namespace pybind11 __attribute__((visibility("hidden")))
{
class array;
}*/

namespace matrix_ops
{
void interpolate_zeros(pybind11::array_t<int> & matrix);
}

