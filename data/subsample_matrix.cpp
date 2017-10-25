// Always include the corresponding header as the first substantive line of code in the source file
#include "subsample_matrix.hpp"
// Then the remainng ones
#include <string.h> // memset

namespace matrix_ops
{

int32_t ** subsample_matrix(const int32_t * const * original_matrix, const int original_matrix_size, const uint32_t desired_matrix_size)
{
	// allocate new matrix
	int32_t ** new_matrix = new int32_t*[desired_matrix_size];
	for (uint32_t i = 0; i < desired_matrix_size; ++i)
	{
		new_matrix[i] = new int32_t[desired_matrix_size];
		memset(new_matrix[i], 0, desired_matrix_size * sizeof(int32_t));
	}

	// populate new matrix
	for (uint32_t row = 0; row < desired_matrix_size; ++row)
	{
		uint32_t zi = (double(row)/desired_matrix_size) * (original_matrix_size - 1);
		for (uint32_t col = 0; col < desired_matrix_size; ++col)
		{
			uint32_t zj = (double(col)/desired_matrix_size) * (original_matrix_size - 1);
			new_matrix[row][col] = original_matrix[zi][zj];
		}
	}

	return new_matrix;
}

} // close namespace matrix_ops
