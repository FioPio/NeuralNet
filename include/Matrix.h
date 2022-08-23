#pragma once // Sols afegeix una vegada l'arxiu en la compilació actual
#include <vector>

class Matrix
{
public:
    int _number_of_rows;
    int _number_of_columns;
    std::vector<float> _values;
public:
    Matrix (int num_rows, int num_cols);
    float & at (int row, int col);
    Matrix multiply (Matrix & matrix_to_multiply);
    Matrix add ( Matrix & matrix_to_add);
    Matrix subtract ( Matrix & matrix_to_add);
};
