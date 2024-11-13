
#include<iostream> 
#include<string> 
#include<tuple> 
#include<vector> 
#include<random> 
#include <fstream>
#include <regex>
#include <sstream>
#include<iomanip> 
#include<algorithm>
#include<cmath> 
#include<limits>
#include <sys/stat.h>  // For checking file existence

#include"layer.h"
#include "../../../flowstar-toolbox/Matrix.h"

using namespace flowstar; 
using namespace std; 


Layer::Layer( Matrix<double>weights_, Matrix<double> bias_, int sigma_){ 
    //weights_ : always passed 
    //bias_ : always passed    
    //sigma : activation function

    weights = weights_ ;
    bias = bias_ ; 
    sigma = sigma_ ;             
}    
  
Matrix<double> Layer::Relu(Matrix<double> a){ 
    for(int i = 0 ; i < a.rows(); i++){
        for(int j = 0; j< a.cols(); j++){
            if(a[i][j] < 0){ // saves a lot of operation w/ < 
                a[i][j] = 0; 
            }
        }
    }
    
    return a; 
}

Matrix<double> Layer::Identity(Matrix<double> a){
    return a ; 
}

// Implement tanh and sigmoid 
Matrix<double> Layer::Tanh(Matrix<double> a){  
    for(int i = 0 ; i < a.rows(); i++){
        for(int j = 0; j< a.cols(); j++){
            a[i][j] = tanh(a[i][j]); 
        }
    }
    return a;  
}
        
Matrix<double> Layer::Sigmoid(Matrix<double> a){
    for(int i = 0 ; i < a.rows(); i++){
        for(int j = 0; j< a.cols(); j++){
            a[i][j] = 1.0 / (1.0 + exp(-a[i][j])); // sigmoid equation 
        }
    }
        
    return a;
} 

Matrix<double> Layer::Softmax(Matrix<double> a, bool return_proba){
    // 1. get the exp of each indices 
    //cout<< "Before applying exp\n"; 
    //cout<< a << endl; 
    double sum = 0.0 ; 
        
    for (int i = 0; i < a.rows(); ++i) {
        a[i][0] = exp( a[i][0] ) ; 
        sum= sum + a[i][0] ; 
    }

    // 2. get the sum of ( exp of all indices) 
    for (int i = 0; i < a.rows(); ++i) {
        a[i][0] = a[i][0] / sum ; 
    } 
    
    return a ; 
}

Matrix<double> Layer::forward(Matrix<double> X_,  bool transpose){
    
    // Perform matrix multiplication                 
    Matrix<double> a; 
    a  = weights * X_  + bias;  //preactivation output 

    if(sigma == 0){
        // This step is redundant; keeping the conditional statement for undertanding 
        a = Identity(a); // redundant!!! 
    }

    else if(sigma == 1){
        a = Relu(a); 
    } 

    else if(sigma == 13){
        a = Softmax(a) ; 
    }

    else{
        cout << "ERROR: Not a valid activation function.\n"; 
    }

    cout << "a shape, after linear operation : (" << a.rows() << ", " << a.cols() << ") " << endl; 

    //cout << "Final value in layer after operation : " << a << endl;
    cout << "******* Layer FORWARD  >>> ENDS!\n";
    
    return a; 
}

void Layer::display_layer(bool displayValues){
        
    cout << "----------------------------" << endl;
    cout << "\nWeights (in layer): (" << weights.size1 << ", " << weights.size2 << ")" << endl;
        
    //weights.rows(); 
    cout << fixed << setprecision(10) << endl; 
    if(displayValues){
        for(int i = 0; i < weights.rows(); i++){
            for(int j = 0; j < weights.cols() ; j++){
                cout << weights[i][j] << " " ;                         
            }cout << endl; 
        }
    }
    
    cout << "----------------------------" << endl; 
    cout << "\nBias (in layer): (" << bias.size1 << ", " << bias.size2 << ")" << endl;
        
    if(displayValues){
        for(int i = 0; i < bias.rows() ; i++){
            cout << bias[i][0] << endl; ; 
        } cout << endl;
    }
        
    cout << "----------------------------" << endl;
    cout << "\nActivation code : " << sigma << endl; 
    cout << "============================" << endl;    
}