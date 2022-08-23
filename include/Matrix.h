#pragma once // Sols afegeix una vegada l'arxiu en la compilació actual

#include <vector>
#include <string>
#include <cassert>
#include <functional>

class Matrix
{
public:
    int _number_of_rows;
    int _number_of_columns;
    std::vector<float> _values;
public:
    Matrix (int num_rows = 0, int num_cols = 0);
    float & at (int row, int col);
    Matrix multiply (Matrix & matrix_to_multiply);
    Matrix hadamardProduct ( Matrix & matrix_to_multiply);
    Matrix add ( Matrix & matrix_to_add);
    Matrix subtract ( Matrix & matrix_to_add);
    Matrix addScalar ( float scalar_to_add);
    Matrix multiplyByScalar ( float multiplying_scalar);
    Matrix negative ();
    Matrix transpose ();
    Matrix applyFunction ( std::function <float (const float &)> function_to_apply);
};
