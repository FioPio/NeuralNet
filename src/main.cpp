#include "NeuralNet.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <iomanip>


void runTests();

int  main()
{
    printf("\n[INFO] Starting Main program.\n");
    
    runTests();
    
    return 0;
}


void runTests()
{
    printf("\n[INFO] Starting tests on Matrix.\n");
    Matrix test_matrix(3,4);
    test_matrix.at(1,2) = 5;
    
    printf(" -> The matrix has %d rows and %d cols, with the value %0.2f at (1, 2).\n",test_matrix._number_of_rows, test_matrix._number_of_columns, test_matrix.at(1,2));
    printf(" -> The matrix has %d rows and %d cols, with the value %0.2f at (0, 2).\n",test_matrix._number_of_rows, test_matrix._number_of_columns, test_matrix.at(0,2));
    
    
    printf("\n[INFO] Starting tests on NeuralNet.\n");
    // Input  layer has 2 neurons
    // Hiden  layer has 3 neurons
    // Output layer has 1 neuron
    std::vector <LayerConfig> net_topology = { 
        LayerConfig( 2), // Input  layer
        LayerConfig( 3, ActivationFunction::ACTIVATION::SIGMOID), // Hidden layer
        LayerConfig( 1, ActivationFunction::ACTIVATION::SIGMOID)  // Output layer
        
    };
    
    NeuralNet nn(net_topology);
        
    std::vector<std::vector<float>> input_data {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}    
    };
    
    // If input different, output is 1, else is 0
    std::vector<std::vector<float>> output_data {
        {0.0f},
        {0.0f},
        {1.0f},
        {1.0f}    
    };
    int epochs = 100000;
    int batch = 4;
    
    nn.train(input_data, output_data, epochs, batch );
    
    printf(" -> Testing the model.\n");
    printf("    -> Final accuracy: %0.2f.\n", nn.test(input_data, output_data));
    
    
    
}
