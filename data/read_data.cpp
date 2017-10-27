#include "read_data.hpp"
#include <pybind11/numpy.h>
#include <fstream>

namespace utils
{

// Read the data from the specified file into a numpy array.
// The raison-d'etre for this function is to provide a mechanism
// to load the data from disk into memory as fast as possible. 
// By writing it in C++ and directly populating a Python datatype,
// speed was improved almost 10x as compared to reading from Python
pybind11::array * read_data(const std::string & data_filename)
{
  std::ifstream data_file;
  data_file.open(data_filename.c_str(), std::ios::in | std::ios::binary);
  int rows = -1;
  data_file.read(reinterpret_cast<char*>(&rows), 4);
  int cols = -1;
  data_file.read(reinterpret_cast<char*>(&cols), 4);
  pybind11::detail::any_container<ssize_t> shape;
  shape->push_back(rows);
  shape->push_back(cols);
  pybind11::detail::any_container<ssize_t> stride;
  pybind11::array * rval = new pybind11::array(pybind11::dtype::of<int>(), shape, stride);
  for(int i = 0; i < rows; ++i)
  {
    for(int j = 0; j < cols; ++j)
    {
      int val = 0;
      data_file.read(reinterpret_cast<char*>(&val), 4);
      int * data = static_cast<int*>(rval->mutable_data(i, j));
      *data = val;
    }
  }
  return rval;
}

}

