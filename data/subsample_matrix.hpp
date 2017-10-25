#pragma once

#include <stdint.h>

namespace matrix_ops
{
	/** @brief: Take a matrix and create a new matrix subsampled down to the specified size. Input is expected to
	            be a square matrix, and output will be a square matrix. CALLER IS RESPONSIBLE FOR DEALLOCATING MEMORY OF RETURN VALUE!
	            Deallocate as follows:
	            for(int i = 0; i < desired_matrix_size; ++i)
	            {
	            	delete[] returned_matrix[i];
	            }
	            delete[] returned_matrix;
	*/
	int32_t ** subsample_matrix(const int32_t * const * original_matrix, const int original_matrix_size, const uint32_t desired_matrix_size);
} // close namespace matrix_ops
