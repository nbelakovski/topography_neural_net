#pragma once

#include <string>

// forward declarations (the "hidden" attribute is to suppress compiler warnings, similar to how pybind11 does it in detail/common.h
namespace pybind11 __attribute__((visibility("hidden")))
{
class array;
}

namespace utils
{
pybind11::array * read_data(const std::string & data_filename);
}

