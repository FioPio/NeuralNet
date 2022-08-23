#include "Matrix.h"
#include <vector>
#include <cassert>

// Constructor of the Class Matrix, generates an empty matrix
Matrix::Matrix (int num_rows, int num_cols)
{
    _number_of_rows = num_rows;
    _number_of_columns = num_cols;
    _values.resize(num_rows * num_cols, 0.0f);    
}

// To obtain the value in one given position, but you can also asign it
float & Matrix::at (int row, int col)
{
    return _values[row*_number_of_columns + col];
}

// Multiply a matrix with another
Matrix Matrix::multiply( Matrix & matrix_to_multiply)
{
    assert (_number_of_columns == matrix_to_multiply._number_of_rows);
    Matrix resulting_matrix (_number_of_rows, matrix_to_multiply._number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            float cell_result = 0;
            for (int num_multiplying_element = 0; num_multiplying_element < _number_of_columns; num_multiplying_element++)
            {
                cell_result += at(num_row, num_multiplying_element) * matrix_to_multiply.at(num_multiplying_element, num_col);
            }
            resulting_matrix.at( num_row, num_col) = cell_result;
        }
    }
    return resulting_matrix;
}

// Adds the matrix with another
Matrix Matrix::add ( Matrix & matrix_to_add)
{
    assert(_number_of_rows == matrix_to_add._number_of_rows && _number_of_columns == matrix_to_add._number_of_columns);
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) - matrix_to_add.at(num_row, num_col);
        }
    }
    return resulting_matrix;
}

// Subtracts the matrix to the current one
Matrix Matrix::subtract ( Matrix & matrix_to_subtract)
{
    assert(_number_of_rows == matrix_to_subtract._number_of_rows && _number_of_columns == matrix_to_subtract._number_of_columns);
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) - matrix_to_subtract.at(num_row, num_col);
        }
    }
    return resulting_matrix;
}
