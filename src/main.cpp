#include "Matrix.h"
#include <stdio.h>

int  main()
{
    printf("\n[INFO] Starting Main code.\n");
    Matrix test_matrix(3,4);
    test_matrix.at(1,2) = 5;
    
    printf("[INFO] Starting tests:\n");
    printf(" -> The matrix has %d rows and %d cols, with the value %0.2f at (2, 3).\n",test_matrix._number_of_rows, test_matrix._number_of_columns, test_matrix.at(1,2));
    printf(" -> The matrix has %d rows and %d cols, with the value %0.2f at (1, 3).\n",test_matrix._number_of_rows, test_matrix._number_of_columns, test_matrix.at(0,2));
    return 0;
}
