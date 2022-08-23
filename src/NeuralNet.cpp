#include "NeuralNet.h"

// Constructor of the class
NeuralNet::NeuralNet(std::vector <LayerConfig> topology, float learning_rate )
{
    _net_topology = topology;
    _net_weights = {};
    _net_values = {};
    _net_bias = {};
    
    _learning_rate = learning_rate;
    
    
    // Create the weight matrices
    for (int num_layer = 0; num_layer < _net_topology.size()-1 ;  num_layer++)
    {
        Matrix layer_weights( _net_topology[num_layer].number_of_neurons, _net_topology[num_layer+1].number_of_neurons);
        // Gives it random initial values
        layer_weights = layer_weights.applyFunction(
            [] ( const float& f )
            {
                return (float) rand() / RAND_MAX;                
            }
        );
        // Adds them into the vector
        _net_weights.push_back(layer_weights);
        
        // Creates bias as a row matrix
        Matrix layer_bias(1,  _net_topology[num_layer+1].number_of_neurons);
        // Gives it random initial values
        layer_bias = layer_bias.applyFunction(
            [] ( const float& f )
            {
                return (float) rand() / RAND_MAX;                
            }
        );
        _net_bias.push_back(layer_bias);
        
        _net_values.resize(_net_topology.size());
    }
}


bool NeuralNet::feedForward(std::vector<float> input_data)
{
    // If wrong size input is provided
    if ( input_data.size() != _net_topology[0].number_of_neurons )
    {
        printf("[ERROR] NeuralNet.feedForward: Expected input data to be %d long, recieved %d\n",_net_topology[0].number_of_neurons, (int) input_data.size());
        return false;
    }
    
    Matrix values ( 1, input_data.size());
    
    for ( int num_input_neuron = 0; num_input_neuron < input_data.size(); num_input_neuron++ )
    {
        values.at(0,num_input_neuron) = input_data[num_input_neuron];
    }
    
    // Feed forward to next layer
    for ( int num_layer_connection = 0; num_layer_connection < _net_weights.size(); num_layer_connection++ )
    {
        _net_values[num_layer_connection] = values;
        
        values = values.multiply(_net_weights[num_layer_connection]);
        values = values.add(_net_bias[num_layer_connection]);
        if (_net_topology[num_layer_connection+1].activation_function==ActivationFunction::ACTIVATION::SIGMOID)
            values = values.applyFunction(ActivationFunction::sigmoid);
        else if (_net_topology[num_layer_connection+1].activation_function==ActivationFunction::ACTIVATION::RELU)
            values = values.applyFunction(ActivationFunction::relu);
    }
    
    _net_values[_net_values.size()-1] = values;
    
    return true;
} 

bool NeuralNet::backPropagate(std::vector<float> output_data)
{
    if ( output_data.size() != _net_topology.back().number_of_neurons )
    {
        printf("[ERROR] NeuralNet.backPropagate: Expected output data to be %d long, recieved %d\n",_net_topology.back().number_of_neurons, (int) output_data.size());
        return false;
    }
    
    // Computing error as : errors = output_data - prediction
    Matrix errors(1, output_data.size());
    errors._values = output_data;
    errors = errors.subtract(_net_values.back());
    
    for (int num_layer_connection = _net_weights.size() -1 ; num_layer_connection >= 0 ; num_layer_connection--)
    {
        Matrix transposed_layer_weights = _net_weights[num_layer_connection].transpose();
        Matrix previous_error = errors.multiply(transposed_layer_weights);
        Matrix output_derivative;
        if (_net_topology[num_layer_connection+1].activation_function==ActivationFunction::ACTIVATION::SIGMOID)
            output_derivative = _net_values[num_layer_connection+1].applyFunction(ActivationFunction::DSigmoid);
        else 
            output_derivative = _net_values[num_layer_connection+1].applyFunction(ActivationFunction::DRelu);
        
        Matrix gradients = errors.hadamardProduct(output_derivative);
        gradients = gradients.multiplyByScalar(_learning_rate);
        
        Matrix weight_gradients = _net_values[num_layer_connection].transpose().multiply(gradients);
        _net_weights[num_layer_connection] = _net_weights[num_layer_connection].add(weight_gradients);
        _net_bias[num_layer_connection] = _net_bias[num_layer_connection].add(gradients);
        errors = previous_error;
        
    }
    
    return true;
}

// Predicts based in one input
std::vector <float> NeuralNet::predict(std::vector <float> input_data)
{
    if ( input_data.size() != _net_topology[0].number_of_neurons )
    {
        printf("[ERROR] NeuralNet.predict: Expected input data to be %d long, recieved %d\n",_net_topology[0].number_of_neurons, (int) input_data.size());
        return {};
    }
    feedForward(input_data);
    
    return _net_values.back()._values;
    
}


// FALTA IMPLEMENTAR EARLY STOP I  BATCH SIZE!
int NeuralNet::train(std::vector<std::vector<float>> input_data, std::vector<std::vector<float>> output_data, int number_of_epochs, int number_of_batchs,  std::vector<std::vector<float>> validate_input_data, std::vector<std::vector<float>> validate_output_data)
{
    bool validate = true;
    if (validate_input_data.size() == 0 || validate_output_data .size() == 0)
        validate = false;
    
    printf("[INFO] Starting training the model.\n");

    // Getting initial time
    clock_t start_time = clock();
    clock_t end_time;
    float elapsed_time = 0;
    float loss = 0; // Es calcula per batch
    // Setting float to be displayed as 0.2f
    std::cout << std::setprecision(2) << std::fixed;
    for (int num_epoch = 0; num_epoch < number_of_epochs; num_epoch++ )
    {
        if (num_epoch % number_of_batchs == 0)
        {
            loss = 0;
        }
        int index = rand() % 4;
        feedForward(input_data[index]);
        backPropagate(output_data[index]);
        float square_error = 0;
        for (int num_output = 0; num_output < output_data[index].size(); num_output++)
        {
            float error = output_data[index][num_output] - _net_values.back()._values[num_output];
            square_error+= error * error;
        }
        
        loss += square_error / ((int) output_data[index].size());
        
        end_time = clock();
        
        float part_done = (((float) num_epoch + 1) / number_of_epochs );
        elapsed_time = float(end_time - start_time)/CLOCKS_PER_SEC;
        
        std::cout << "\r[" << num_epoch + 1 << "/" << number_of_epochs  << "] " << 100 * part_done << "%. Loss: "<< loss*1000 <<" Elapsed: "<< elapsed_time <<"s, Total: "<<elapsed_time/part_done  <<"s." << std::flush;
    }
    printf("\n");
    printf("    -> Done in %0.2fs.\n", elapsed_time);
    return number_of_epochs;
}

float NeuralNet::test(std::vector<std::vector<float>> input_data, std::vector<std::vector<float>> output_data, float tolerance )
{
    int num_correct_predictions = 0;
    int predictions = 0;
    for (int num_input = 0; num_input < input_data.size(); num_input++ )
    {
        std::vector<float> predicted_output = predict(input_data[num_input]);
        for ( int num_output = 0; num_output < output_data[num_input].size(); num_output++)
        {
            float error = output_data[num_input][0] - predicted_output[0];
        
            if ( error < tolerance && error > -tolerance )
            {
                num_correct_predictions++;
            }
            predictions++;
        }
        
        //printf("     -> [%02d/%02d] Expected %0.2f, obtained %0.2f.\n", num_input + 1, (int) input_data.size(), output_data[num_input][0], predicted_output[0]);
    }
    
    return (( float ) num_correct_predictions )/ predictions;
}



float ActivationFunction::sigmoid ( float x )
{
    return 1.0/( 1 + exp( -x ) );
}

float ActivationFunction::DSigmoid ( float x )
{
    return x * ( 1 - sigmoid(x));
}

float ActivationFunction::relu ( float x)
{
    if (x>0)
        return x;
    return 0;
}
float ActivationFunction::DRelu ( float x)
{
    if (x>0)
        return 1;
    return 0;
}

LayerConfig::LayerConfig(int n_neurons, ActivationFunction::ACTIVATION activ_function)
{
    number_of_neurons = n_neurons;
    activation_function = activ_function;
}
