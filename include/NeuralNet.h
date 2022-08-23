#pragma once

#include "Matrix.h"
#include <string>
#include <vector>
#include <cassert>
#include <functional>
#include <cstdlib>
#include <bits/stdc++.h>
#include <time.h>
#include <iomanip>


namespace ActivationFunction
{
    float sigmoid ( float x );
    float DSigmoid ( float x );
    float relu ( float x);
    float DRelu ( float x);
    
    enum ACTIVATION 
    {
        SIGMOID, 
        RELU
    };
};


struct  LayerConfig
{
    int number_of_neurons;
    ActivationFunction::ACTIVATION activation_function;
    LayerConfig(int n_neurons, ActivationFunction::ACTIVATION activ_function = ActivationFunction::ACTIVATION::SIGMOID);
};


class NeuralNet
{
    std::vector<LayerConfig> _net_topology;    // Each value is a layer, and the number represents the total neurons the layer has.
    std::vector<Matrix> _net_weights;  // A vector that contains the weights for each layer connection.
    
    std::vector<Matrix> _net_bias;     // A vector containing the bias for each connection.
    
    float _learning_rate;
    
public:
    std::vector<Matrix> _net_values;   // Contains the values of the input on each layer.
    
public:
    NeuralNet(std::vector <LayerConfig> topology, float learing_rate = 0.1f);
    void save(std::string path_to_save);
    void load(std::string path_to_load);
    bool feedForward(std::vector<float> input_data);
    bool backPropagate(std::vector<float> output_data);
    std::vector <float> predict(std::vector <float> input_data);
    int train(std::vector<std::vector<float>> input_data, std::vector<std::vector<float>> output_data, int number_of_epochs, int number_of_batchs, std::vector<std::vector<float>> validate_input_data={}, std::vector<std::vector<float>> validate_output_data={});
    float test(std::vector<std::vector<float>> input_data, std::vector<std::vector<float>> output_data, float tolerance = 0.05);
};

