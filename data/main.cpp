#include "convert_las_to_matrix.hpp"
#include <iostream>
#include <string>

using namespace std;

const int DESIRED_MATRIX_SIZE = 500;

int main(int argc, char const *argv[])
{
	if (argc < 2)
    {
        cout << "Usage: ./" << argv[0] << " [las_filename]" << endl;
        return 1;
    }

    const string las_filename(argv[1]);
	int32_t ** subsampled_matrix = converter::convert_las_to_matrix(las_filename, DESIRED_MATRIX_SIZE);

	// write output to file
	std::string out_filename(las_filename.substr(0, las_filename.size() - 3) + "data");
	converter::write_matrix_to_file(out_filename, subsampled_matrix, DESIRED_MATRIX_SIZE);

	// deallocate memory
	for (int i = 0; i < DESIRED_MATRIX_SIZE; ++i)
    {
        delete[] subsampled_matrix[i];
    }
    delete[] subsampled_matrix;

    return 0;
}