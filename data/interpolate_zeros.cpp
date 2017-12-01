#include "interpolate_zeros.hpp"
#include <pybind11/numpy.h>

namespace matrix_ops
{

int interpolate_zeros(pybind11::array_t<int> & matrix)
{
  int rows = matrix.shape()[0];
  int cols = matrix.shape()[1];
  for (int row = 0; row < rows; ++row)
  {
    int start_col = -1;
    int end_col = -1;
    for (int col = 0; col < cols; ++col)
    {
      const int * data = static_cast<const int*>(matrix.data(row, col));
      if (start_col == -1 && *data == 0)
      {
        start_col = col;
      }
      if (start_col != -1 && end_col == -1 && *data != 0)
      {
        end_col = col; // this might not get reached if line ends with a 0
      }

      // if end_col is -1, it could mean a 0 at the end, meaning we should
      // also check to see if we are at the end
      if (start_col != -1 && ((end_col != -1) || col == cols - 1))
      {
        int start_val = 0;
        int end_val = 0;
        if (start_col == 0 && end_col == -1)
        {
          return -1;
        }
        else if (start_col == 0)
        {
          start_val = end_val = *static_cast<const int*>(matrix.data(row, end_col));
        }
        else if (end_col == -1)
        {
          end_col = cols;
          start_val = end_val = *static_cast<const int*>(matrix.data(row, start_col - 1));
        }
        else
        {
          start_val = *static_cast<const int*>(matrix.data(row, start_col - 1));
          end_val   = *static_cast<const int*>(matrix.data(row, end_col));
        }
        float slope = float(end_val - start_val) / float(end_col - start_col + 1);
        int intercept = start_val; // even in the case of a line starting with a 0, this is ok, since start_val == end_val in that case
        for (int i = start_col; i < end_col; ++i)
        {
          int * data = static_cast<int*>(matrix.mutable_data(row, i));
          *data = slope * (i - start_col + 1) + intercept;
        }
        start_col = end_col = -1;
      }
    }
  }
  return 0;
}

}
