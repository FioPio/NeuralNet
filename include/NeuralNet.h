#pragma once

#include "Matrix.h"

class NeuralNet
{
    std::vector<int> _net_topology;
    std::vector<Matrix> _net_weights;
    std::vector<Matrix> _net_values; //?
    std::vector<Matrix> _net_bias; 
public:
    void save(std::string path_to_save);
    void load(std::string path_to_load);
}
