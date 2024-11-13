#ifndef LAYER_H
#define LAYER_H

#include <iostream>    // For std::cout
#include <cmath>       // For tanh, exp
#include <fstream>     // For file handling
#include <regex>       // For regex operations
#include <algorithm>
#include "layer.h"

#include "../../../flowstar-toolbox/Matrix.h" // For Matrix<double>

using namespace flowstar; 
using namespace std; 

class Layer{
    
    private:    
    
    Matrix<double> weights;   
    Matrix<double> bias;
    int sigma;  // activation function  
      
    public: 
    
    Layer( Matrix<double>weights_, Matrix<double> bias_, int sigma_);    
  
    Matrix<double> Relu(Matrix<double> a);
    Matrix<double> Identity(Matrix<double> a); 
    Matrix<double> Tanh(Matrix<double> a);        
    Matrix<double> Sigmoid(Matrix<double> a); 
    Matrix<double> Softmax(Matrix<double> a, bool return_proba= true);
    Matrix<double> forward(Matrix<double> X_,  bool transpose=false); 
    void display_layer(bool displayValues=false); 

}; 


#endif // LAYER_H