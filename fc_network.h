#include<iostream>
#include "../../../flowstar-toolbox/Matrix.h"
#include "layer.h"

using namespace flowstar; 
using namespace std;

class FCNetwork{

    private: 
        // A vector of Layer type 
        vector<Layer> network ;         
        
    public: 

        FCNetwork(const string &filename);
        Matrix<double> forward( Matrix<double> X_);
        void display_network();
};