#include "interpolate_zeros.hpp"
#include <pybind11/numpy.h>
#include <iostream>

namespace matrix_ops
{

void interpolate_zeros(pybind11::array_t<int> & matrix)
{
  /*pybind11::buffer_info b = matrix.request();
  void * const ptr = b.ptr;  // make it const so that we have the initial location always stored
  int * data = static_cast<int*>(ptr);
  *data = 56;
  std::cout << *(static_cast<const int*>(ptr) + 4) << std::endl;*/
  std::cout << *(static_cast<int*>(matrix.mutable_data(0,0))) << std::endl;
  int * data = static_cast<int*>(matrix.mutable_data(0,1));
  *data = 78;
  std::cout << matrix.shape()[0] << std::endl;
  std::cout << matrix.shape()[1] << std::endl;
}

}
