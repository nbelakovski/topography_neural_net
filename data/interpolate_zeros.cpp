#include "interpolate_zeros.hpp"
#include <pybind11/numpy.h>
#include <iostream>

namespace matrix_ops
{
int find_nonzero_vals(const pybind11::array_t<int> & matrix, const int row, const int col)
{
  // given a data point whose value is 0, look up and down the rows for the first nonzero value
  // return the closest one
  const int rows = matrix.shape()[0];
  // Look up first
  int up_distance = -1;
  for(int i = row; i > 0; --i)
  {
    const int * data = static_cast<const int*>(matrix.data(i, col));
    if(*data != 0)
    {
        up_distance = (row - i);
        break;
    }
  }
  // then look down (but don't bother looking further than up_distance, if it's valid
  // so, if down_distance is valid, it's guaranteed to be <= up_distance
  int down_distance = -1;
  for(int i = row; i < rows || (up_distance != -1 && i < (row + up_distance)); ++i)
  {
    const int * data = static_cast<const int*>(matrix.data(i, col));
    if(*data != 0)
    {
        down_distance = (i - row);
        break;
    }
  }
  if (down_distance != -1)
  {
      return *static_cast<const int*>(matrix.data(row + down_distance, col));
  }
  else if (up_distance != -1)
  {
      return *static_cast<const int*>(matrix.data(row - up_distance, col));
  }
  return 0;
}

int interpolate_zeros(pybind11::array_t<int> & matrix)
{
  const int rows = matrix.shape()[0];
  const int cols = matrix.shape()[1];
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
          // this means the entire row is blank.
          // the previous row should be copied
          // if this is the first row, it should be marked and filled when interpolation
          // of the remainder of the matrix is complete
          return -1;
        }
        else if (start_col == 0)
        {
          // look up and down the rows for the nearest non-zero value
          // if none is found, mark the row, and continue interpolating the rest of the matrix
          start_val = find_nonzero_vals(matrix, row, 0);
          end_val = *static_cast<const int*>(matrix.data(row, end_col));
          if (start_val == 0) { start_val = end_val; }
        }
        else if (end_col == -1)
        {
          end_col = cols;
          start_val = *static_cast<const int*>(matrix.data(row, start_col - 1));
          end_val = find_nonzero_vals(matrix, row, cols - 1);
          if (end_val == 0) { end_val = start_val; }
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

void add_to_value_and_total(const pybind11::array_t<int> & matrix, const int row, const int col, int * value, int * total)
{
   int val = *static_cast<const int*>(matrix.data(row, col));
   if (val != 0)
   {
       *value += val;
       *total += 1;
   }
}

int populate_index_from_neighbors(pybind11::array_t<int> & matrix, const std::pair<int, int> & index)
{
   int total = 0;
   int value = 0;
   const int rows = matrix.shape()[0];
   const int cols = matrix.shape()[1];
   const int row = index.first;
   const int col = index.second;
   for(int i = std::max(row - 5, 0); i < std::min(row + 6, rows); ++i)
   {
       for(int j = std::max(col - 5, 0); j < std::min(col + 6, cols); ++j)
       {
           if(i == row && j == col) {continue;}
           add_to_value_and_total(matrix, i, j, &value, &total);
       }
   }

   if(total > 0)
   {
       int * data = static_cast<int*>(matrix.mutable_data(row, col));
       *data = value / total;
       return 0;
   }
   else
   {
       return -1;
   }
}

void check_zero_and_add_to_list(const pybind11::array_t<int> & matrix, const int row, const int col, std::vector<std::pair<int, int>> & zero_value_indices)
{
    int val = *static_cast<const int*>(matrix.data(row, col));
    if (val == 0)
    {
        std::pair<int, int> temp(row, col);
        zero_value_indices.push_back(temp);
    }
}

// Alternate algorithm. Find all 0's, put them in a list, go through them and populate all that have non-zero neighbors, and do this recursively until list is empty
void interpolate_zeros_2(pybind11::array_t<int> & matrix)
{
  const int rows = matrix.shape()[0];
  const int cols = matrix.shape()[1];
  std::vector<std::pair<int, int>> zero_value_indices;
  // Identify all indices with a value of 0
  /* for(int row = 0; row < rows; ++row) */
  /* { */
  /*     for(int col = 0; col < cols; ++col) */
  /*     { */
  /*         check_zero_and_add_to_list(matrix, row, col, zero_value_indices); */
  /*     } */
  /* } */
  // Idea: build out the zero index from a spiral starting in the middle and moving outwards. Having control over the order might improve the result
  int pivot_row = rows / 2;
  int pivot_col = cols / 2;
  int stride = 1;
  bool top_left_corner_hit = false;
  bool top_right_corner_hit = false;
  bool bottom_left_corner_hit = false;
  bool bottom_right_corner_hit = false;
  while(!top_left_corner_hit && !top_right_corner_hit && !bottom_left_corner_hit && !bottom_right_corner_hit)
  {
      // one spiral is up stride, right stride, down stride+1, left stride+1
      int next_row = pivot_row - stride;
      int next_col = pivot_col;
      for(int i = std::min(pivot_row, rows - 1); i >= next_row && i >= 0 && pivot_col >= 0 && pivot_col < cols; --i)
      {
          check_zero_and_add_to_list(matrix, i, pivot_col, zero_value_indices);
      }
      if(next_col <= 0 && next_row <= 0) {top_left_corner_hit = true;}
      pivot_row = next_row;
      pivot_col = next_col;
      //right stride
      next_row = pivot_row;
      next_col = pivot_col + stride;
      for(int i = std::max(pivot_col, 0); i <= next_col && i < cols && pivot_row >= 0 && pivot_row < rows; ++i)
      {
          /* std::cout << i << std::endl; */
          check_zero_and_add_to_list(matrix, pivot_row, i, zero_value_indices);
      }
      if(next_col >= cols && next_row <= 0) {top_right_corner_hit = true;}
      pivot_row = next_row;
      pivot_col = next_col;
      // down stride
      stride += 1;
      next_row = pivot_row + stride;
      next_col = pivot_col;
      for(int i = std::max(pivot_row, 0); i <= next_row && i < rows && pivot_col >= 0 && pivot_col < cols; ++i)
      {
          check_zero_and_add_to_list(matrix, i, pivot_col, zero_value_indices);
      }
      if(next_col >= cols && next_row >= rows) {bottom_right_corner_hit = true;}
      pivot_row = next_row;
      pivot_col = next_col;
      // left stride
      next_row = pivot_row;
      next_col = pivot_col - stride;
      for(int i = std::min(pivot_col, cols - 1); i >= next_col && i >= 0 && pivot_row >= 0 && pivot_row < rows; --i)
      {
          /* std::cout << i << ", " << pivot_row << std::endl; */
          check_zero_and_add_to_list(matrix, pivot_row, i, zero_value_indices);
      }
      if(next_col <= 0 && next_row >= rows) {bottom_left_corner_hit = true;}
      pivot_row = next_row;
      pivot_col = next_col;
      stride += 1;
  }

  // Now iterate through all of these values, and for each value, populate it with the average of its non-zero neighbors, and remove it from the list
  // If it has no non-zero neighbors, skip it
  while (zero_value_indices.size() > 0)
  {
      for(std::vector<std::pair<int, int>>::iterator index = zero_value_indices.begin(); index != zero_value_indices.end();)
      {
          int rcode = populate_index_from_neighbors(matrix, *index);
          if (rcode == 0) // 0 means success
          {
              index = zero_value_indices.erase(index);
          }
          else
          {
              ++index;
          }
      }
  }
}

}
