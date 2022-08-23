#include "Matrix.h"


// Constructor of the Class Matrix, generates an empty matrix
Matrix::Matrix (int num_rows, int num_cols)
{
    _number_of_rows = num_rows;
    _number_of_columns = num_cols;
    if ( num_rows > 0 && num_cols > 0)
    {
         _values.resize(num_rows * num_cols, 0.0f); 
    }
    
}

// To obtain the value in one given position, but you can also asign it
float & Matrix::at (int row, int col)
{
    return _values[row*_number_of_columns + col];
}

// Multiply a matrix by another
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

// Performs the Hadamard product (a[i,j] = b[i,j] * c[i,j])
Matrix Matrix::hadamardProduct ( Matrix & matrix_to_multiply)
{
    assert (_number_of_columns == matrix_to_multiply._number_of_columns && _number_of_rows == matrix_to_multiply._number_of_rows );
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) * matrix_to_multiply.at(num_row, num_col);
        }
    }
    return resulting_matrix;
}

// Adds a matrix to the current one
Matrix Matrix::add ( Matrix & matrix_to_add)
{
    assert(_number_of_rows == matrix_to_add._number_of_rows && _number_of_columns == matrix_to_add._number_of_columns);
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) + matrix_to_add.at(num_row, num_col);
        }
    }
    return resulting_matrix;
}

// Subtracts a matrix to the current one
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

// Adds an scalar to the matrix
Matrix Matrix::addScalar ( float scalar_to_add)
{
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) + scalar_to_add;
        }
    }
    return resulting_matrix;
}

// Multiplies the matrix by an scalar
Matrix Matrix::multiplyByScalar ( float multiplying_scalar)
{
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_row, num_col) * multiplying_scalar;
        }
    }
    return resulting_matrix;
}

// Returns negative the matrix
Matrix Matrix::negative ()
{
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = -at(num_row, num_col);
        }
    }
    return resulting_matrix;
}

// Returns negative the matrix
Matrix Matrix::transpose ()
{
    Matrix resulting_matrix ( _number_of_columns, _number_of_rows);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at( num_row, num_col) = at(num_col, num_row);
        }
    }
    return resulting_matrix;
}

Matrix Matrix::applyFunction ( std::function <float (const float &)> function_to_apply)
{
    Matrix resulting_matrix (_number_of_rows, _number_of_columns);
    for (int num_row=0; num_row < resulting_matrix._number_of_rows;num_row++)
    {
        for (int num_col=0; num_col < resulting_matrix._number_of_columns; num_col++)
        {
            resulting_matrix.at (num_row, num_col) = function_to_apply ( at (num_row, num_col) );
        }
    }
    return resulting_matrix;
}



