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
#include "../../../flowstar-toolbox/Matrix.h"
#include "layer.h"
#include "fc_network.h"

using namespace flowstar; 
using namespace std;

FCNetwork::FCNetwork(const string &filename){
            
    string onnxFile = filename ;
    int numLayers;       // Number of layers
    int layerCount = 0; 
            
    vector<string> fileLines;
    vector<int> weightRows; 
    vector<int> weightCols; 

    vector<int> biasRows; 
    vector<int> biasCols; 

    vector<int> sigmaCode; 

    // The constructor takes in .onnx file as parameter and creates a text file where it puts the weights, biases, and activations             
    cout << "In filename constructor for Net class/object with " << onnxFile << " passed as parameter.\n\n"; 
        
    // Call the bash script that parses the onnx file 
    string command = "./run_onnx_in_python.sh " + onnxFile;
    cout << "Command for shell file execution : " << command << " " << onnxFile ; 
    system(command.c_str()); // works fine 
            
    // Find the position of the last dot
    size_t lastDotPosition = onnxFile.find_last_of('.');

    // Extract the substring before the last dot
    string nameBeforeExtension = onnxFile.substr(0, lastDotPosition);

    // Print the result
    cout << "Name before extension: " << nameBeforeExtension << std::endl;
          
    string folderPath = "weights_and_biases/";
    string fileName = nameBeforeExtension +"_weights_and_biases" + ".txt";
    string filePath = folderPath + fileName ; 
            
    cout << "folderPath : " << folderPath << endl; 
    cout << "fileName : " << fileName << endl; 
    cout << "filePath : " << filePath << endl; 

    ifstream file(filePath);
                        
    cout << "Attempt made to open file.\n"; 
    string line;
            
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
    }

    while (getline(file, line)) {
        if (line.find("Weight Shape:") != string::npos) {
            layerCount++;
        }                
    }
    
    file.close();
            
    cout << "Opened file to count the number of layers and close it\n"; 
    cout << "No. of layers in neural network : " << layerCount << endl; 

    // Initialize the number of layers
    numLayers = layerCount;

    // Reset file pointer and start parsing again
    file.clear();
    file.seekg(0);
    file.close(); 

    int currentLayer = 0;
    ifstream file_1(filePath);
        
    cout << "BEFORE THE BIG TASK\n"; 
        
    while (getline(file_1, line)) {                
        fileLines.push_back(line);                
    }

    file_1.close(); // Close the file stream
       
    // Parse weight shape
    int rows = 0, cols = 0;
    double** weights = nullptr; // Pointer for weight matrix
    double** biases = nullptr;   // Pointer for bias vector

    Matrix<double> weights_fs ; 
    Matrix<double> bias_fs; 
    int sigma = 0;

    bool assignedWeight = false; 
    bool assignedBias = false; 
    bool assignedSigma = false; 

    unsigned lineNo = 0;  

    for (unsigned lineNo = 0; lineNo < fileLines.size(); lineNo++) {
        cout << "------------------------\n";
        cout << fileLines[lineNo] << endl;
        cout << "------------------------\n";

        if (fileLines[lineNo].find("Weight Shape:") != std::string::npos) {
            // Extract the values inside parentheses
            regex rgx("\\((\\d+),\\s*(\\d+)\\)");
            smatch match;

            if (std::regex_search(fileLines[lineNo], match, rgx) && match.size() == 3) {
                rows = stoi(match[1].str());
                cols = stoi(match[2].str());

                cout << "Rows: " << rows << ", Columns: " << cols << '\n';

                //// Declare flowstar matrix 
                Matrix<double> w_ (rows, cols);                         
                        
                // Read and populate the 2D array from the next lines
                for (int currentRow = 0; currentRow < rows && lineNo + 1 < fileLines.size(); ++currentRow) {
                    lineNo++; // Move to the next line

                    //cout << "---> string before erase operation : " << fileLines[lineNo] << endl;
                            
                    // Erase brackets from the line
                            
                    fileLines[lineNo].erase(std::remove(fileLines[lineNo].begin(), fileLines[lineNo].end(), '['), fileLines[lineNo].end());
                    fileLines[lineNo].erase(std::remove(fileLines[lineNo].begin(), fileLines[lineNo].end(), ']'), fileLines[lineNo].end());
                            
                    //cout << "---> string after erase operation : " << fileLines[lineNo] << endl;

                    // Now parse the values into the weights matrix
                    stringstream ss(fileLines[lineNo]);
                    for (int c = 0; c < cols; ++c) {
                        ss >> w_[currentRow][c];
                        if (c < cols - 1) ss.ignore(1); // Ignore comma
                        }
                }   

                // Display the weights matrix after initialization
                cout << "Weights Matrix, after initialization:\n";
                //cout << w_ << endl;
                weights_fs = w_ ;  
                cout << "weights_fs, row : " << weights_fs.size1 << ", col : " << weights_fs.size2 << endl; 
               
                assignedWeight = true;
                // TASK: clear the memory in w_ and weights
                    }
                } // end of parsing weight

                // Parse biases
                if (fileLines[lineNo].find("Bias Shape:") != std::string::npos) {
                    // Extract the number of biases
                    regex rgx("\\((\\d+),\\)");
                    smatch match;
                    
                    if (std::regex_search(fileLines[lineNo], match, rgx) && match.size() == 2) {
                        int biasCount = stoi(match[1].str());  // Extract the number of biases

                        // Declare flowstar matrix with biasCount rows and 1 column
                        Matrix<double> b_(biasCount, 1);

                        // Move to the next line where bias values start
                        lineNo++;
                        
                        // Now parse the bias values, line by line
                        for (int r = 0; r < biasCount; ++r, ++lineNo) {
                            stringstream ss(fileLines[lineNo]);

                            // Parse and store the bias value into the matrix
                            ss >> b_[r][0];
                        }

                        // Display the biases after initialization (optional)
                        cout << "b_ shape: (" << b_.size1 << ", " << b_.size2 << ")\n";

                        // Assign the parsed biases to bias_fs
                        bias_fs = b_;
                        assignedBias = true;
                    }                
                } // end of parsing biases
            
                // Parse Activation Code
                if (fileLines[lineNo].find("Activation Code:") != std::string::npos) {
                    regex rgx("Activation Code: (\\d+)");
                    smatch match;

                    if (std::regex_search(fileLines[lineNo], match, rgx) && match.size() == 2) {
                        int activationCode = stoi(match[1].str()); // Convert to integer
                        sigma = activationCode; // Store activation code
                        cout << "Stored Activation Code: " << sigma << endl; // Display the activation code
                    }

                    assignedSigma = true; 
                } // end of sigma/activation function

                // Create a Layer instance with the data parsed above 
                // At any given iteration, I can read one of weight, bias, and activation/sigma. Here, I need all of them.            

                if(assignedWeight && assignedBias && assignedSigma){
                    //cout << "\t\t\t\tAt iteration  " << lineNo << "##### create a layer object ####\n"; 
                                    
                    // Variables under consideration : weights_fs, bias_fs, sigma 
                    Layer layer(weights_fs, bias_fs, sigma);

                    network.push_back(layer); 
                    
                    assignedWeight = false; 
                    assignedBias = false; 
                    assignedSigma = false; 

                }
            } // end of for (unsigned lineNo = 0; lineNo < fileLines.size(); lineNo++)
          
}    

Matrix<double> FCNetwork::forward( Matrix<double> X_){ // This is the function I am calling 
           
    // Declare an instance of flow* matrix             
    int row = X_.rows(); 
    int col = X_.cols(); 

    Matrix<double> h(row, col) ;
             
    for (int i =0; i < row; i++){
        for(int j = 0; j< col; j++){
            h[i][j] = X_[i][j]; 
            //cout << "("<< i << ", " << j << ")" <<endl; 
        }
    }

    cout << "Inside the forward function for network class\n"; 
    cout << "Dimension of h (aka input/hidden layer): (" << h.rows() << ", " << h.cols() << ") \n";  
    
    for(int i = 0; i<network.size(); i++){
                
        cout << "\t\t\t\t**** Start of iteration at network " << i << endl; 
        h = network[i].forward(h) ;
        cout << "\t\t\t\t**** End of iteration at network " << i << endl;        
    }
    return h;
}

void FCNetwork::display_network(){
    cout << "Inside display_network function\n\n";

    for(size_t i = 0; i< network.size(); i++){
        network[i].display_layer() ; 
    }
} 