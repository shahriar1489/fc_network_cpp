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

using namespace flowstar; 
using namespace std; 

class Layer{
    
    private:    
    Matrix<double> weights; // this matrix stores the weights  
    int sigma;  // activation function  
    Matrix<double> bias; // 
    
    public: 
    Layer( Matrix<double>weights_, Matrix<double> bias_, int sigma_){ 
        //weights_ : always passed 
        //bias_ : always passed    
        //sigma : activation function

        weights = weights_ ;
        bias = bias_ ; 
        sigma = sigma_ ;             
    }    
  
    Matrix<double> Relu(Matrix<double> a){ 
        for(int i = 0 ; i < a.rows(); i++){
            for(int j = 0; j< a.cols(); j++){
                if(a[i][j] < 0){ // saves a lot of operation w/ < 
                    a[i][j] = 0; 
                }
            }
        }
        return a; 
    }

    Matrix<double> Identity(Matrix<double> a){
        return a ; 
    }

    // Implement tanh and sigmoid 
    Matrix<double> Tanh(Matrix<double> a){  
        for(int i = 0 ; i < a.rows(); i++){
            for(int j = 0; j< a.cols(); j++){
                    a[i][j] = tanh(a[i][j]); 
                }
            }
        return a;  
    }
        
   Matrix<double> Sigmoid(Matrix<double> a){
       for(int i = 0 ; i < a.rows(); i++){
            for(int j = 0; j< a.cols(); j++){
                   a[i][j] = 1.0 / (1.0 + exp(-a[i][j])); // sigmoid equation 
                }
            }
        
        return a;
   } 

    Matrix<double> Softmax(Matrix<double> a, bool return_proba= true){
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

    Matrix<double> forward(Matrix<double> X_,  bool transpose=false){
    
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

    void display_layer(bool displayValues=false){
        
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
};

class FCNetwork{

    private: 
        // A vector of Layer type 
        vector<Layer> network ;         
        
    public: 

        FCNetwork(const string &filename){
            
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

        Matrix<double> forward( Matrix<double> X_){ // This is the function I am calling 
           
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

            //exit(-2);            
            for(int i = 0; i<network.size(); i++){
                
                cout << "\t\t\t\t**** Start of iteration at network " << i << endl; 
                h = network[i].forward(h) ;
                cout << "\t\t\t\t**** End of iteration at network " << i << endl;        
            }

            return h; 
        }

        void display_network(){
            cout << "Inside display_network function\n\n";

            for(size_t i = 0; i< network.size(); i++){
                network[i].display_layer() ; 
            }

        }        
};


void testModelsOnSampledData(){
    
    //int modelNo = 1; 
    
    string modelFolder = "model"; 

    string model_01  = "model_01.onnx";    
    string model_02 =  "model_02.onnx";
    string model_03 =  "model_03.onnx";
    string model_04 =  "model_04.onnx";
    string model_05 =  "model_05.onnx";

    //#net net11 = Net(model_01); 
    
    /* 1. I read the ONNX file here; 
    */

    FCNetwork net11(model_01);
    cout << "**** model_01 successfully loaded.\n"; 
    
    FCNetwork net21(model_02); 
    cout << "**** model_02 successfully loaded.\n"; 
    
    FCNetwork net31(model_03); 
    cout << "**** model_04 successfully loaded.\n"; 

    FCNetwork net41(model_04); 
    cout << "**** model_04 successfully loaded.\n"; 

    FCNetwork net51(model_05); 
    cout << "**** model_05 successfully loaded.\n"; 

    //
    /* 2. In the csv file above, I want to I want to avoid reading the first line of the file which is the header
    From the second line, I want to read the first of the two entries as double data type. The entry is to be saved in a vector 
    */
 
    string dataFolder, sampleFileName, dataFilePath; 
    //string modelFolder, modelName, modelFilePath; 
    vector<double> X_test;

    //cout << "Generating results for model_01 \n"; 
    // test for model_01 
    
    //cout <<"Generating results for model_01" << endl;
    for(int i = 1; i <= 5; i++)  {
        dataFolder = "data" ; 
        sampleFileName = "samples_model_0" + to_string(i) ;   
        dataFilePath = dataFolder + "/" + sampleFileName + ".csv" ; 

        cout << "Reading from file : " << dataFilePath << endl; 
        // Read from the second line of the file; 
        // There are two entries per line, read the first  entry only 
        // save the data in double type vector 
   
        // Read from the second line of the file:
        ifstream file(dataFilePath);
        if (!file.is_open()) {
            cerr << "Error opening file: " << dataFilePath << endl;
            return ;
        }

        string line;
        getline(file, line); // Skip the first line

        // Read until the end of the file:
        while (getline(file, line)) {
            // Extract the first entry from the line:
            istringstream iss(line);
            double value;
        
            if (iss >> value) {
                X_test.push_back(value);
                cout << "Push into X_test : " << value << endl; 
            } else {
                cerr << "Error reading value from line: " << line << endl;
            }
        }
         
        Matrix<double> test_tuple_fs(1, 1) ;   
        vector<double> results;
        
        //for(int i=1; i < 6; i++){
        //    modelFolder = "model"; 
        //    modelName = "model_0" + to_string(i); 

            
        cout << "test_tuple_fs : " << test_tuple_fs << endl;         
        for(unsigned j = 0; j < X_test.size() ; j++){
            test_tuple_fs[0][0] = X_test.at(j);
            cout << "test_tuple_fs, after initialization : " << test_tuple_fs << endl; 
            // Pass the tuple to forward function 
            results.push_back ( net11.forward(test_tuple_fs)[0][0] ); // -- >> Change here 
        }

        cout << "--- We have the results for input file : " << sampleFileName << endl; 

        string modelName = "model_01";  // -- >> Change here 

        string resultFilePath = "results/onnx/" + modelName + "_sample_0" + to_string(i) + ".txt"; 

        // Write results vector to file
        ofstream resultFile(resultFilePath);
        if (!resultFile.is_open()) {
            std::cerr << "Error opening file: " << resultFilePath << std::endl;
            return ; // Handle error appropriately if needed
        }
        resultFile << setprecision(16);
        // Write each result to a separate line
        for(unsigned j = 0; j < results.size(); j++){
            resultFile << results[j] << endl; 
        }
        

        resultFile.close();


        X_test.clear(); 
        file.close();
    }

    cout << "Generating results for model_02 " << endl; 

    for(int i = 1; i <= 5; i++)  {
        dataFolder = "data" ; 
        sampleFileName = "samples_model_0" + to_string(i) ;   
        dataFilePath = dataFolder + "/" + sampleFileName + ".csv" ; 

        cout << "Reading from file : " << dataFilePath << endl; 
        // Read from the second line of the file; 
        // There are two entries per line, read the first  entry only 
        // save the data in double type vector 
   
        // Read from the second line of the file:
        ifstream file(dataFilePath);
        if (!file.is_open()) {
            cerr << "Error opening file: " << dataFilePath << endl;
            return ;
        }

        string line;
        getline(file, line); // Skip the first line

        // Read until the end of the file:
        while (getline(file, line)) {
            // Extract the first entry from the line:
            istringstream iss(line);
            double value;
        
            if (iss >> value) {
                X_test.push_back(value);
                cout << "Push into X_test : " << value << endl; 
            } else {
                cerr << "Error reading value from line: " << line << endl;
            }
        }
         
        Matrix<double> test_tuple_fs(1, 1) ;   
        vector<double> results;
            
        cout << "test_tuple_fs : " << test_tuple_fs << endl;         
        for(unsigned j = 0; j < X_test.size() ; j++){
            test_tuple_fs[0][0] = X_test.at(j);
            cout << "test_tuple_fs, after initialization : " << test_tuple_fs << endl; 
            // Pass the tuple to forward function 
            results.push_back ( net21.forward(test_tuple_fs)[0][0] ); // -- >> Change here 
        }

        cout << "--- We have the results for input file : " << sampleFileName << endl; 

        string modelName = "model_02";  // -- >> Change here 

        string resultFilePath = "results/onnx/" + modelName + "_sample_0" + to_string(i) + ".txt"; 

        // Write results vector to file
        ofstream resultFile(resultFilePath);
        if (!resultFile.is_open()) {
            std::cerr << "Error opening file: " << resultFilePath << std::endl;
            return ; // Handle error appropriately if needed
        }
        resultFile << setprecision(16);
        // Write each result to a separate line
        for(unsigned j = 0; j < results.size(); j++){
            resultFile << results[j] << endl; 
        }
        
        resultFile.close();

        X_test.clear(); 
        file.close();

    }

    //print("-------------------------\n"); 
    cout << "Generating results for model_03 " << endl; 

    for(int i = 1; i <= 5; i++)  {
        dataFolder = "data" ; 
        sampleFileName = "samples_model_0" + to_string(i) ;   
        dataFilePath = dataFolder + "/" + sampleFileName + ".csv" ; 

        cout << "Reading from file : " << dataFilePath << endl; 
        // Read from the second line of the file; 
        // There are two entries per line, read the first  entry only 
        // save the data in double type vector 
   
        // Read from the second line of the file:
        ifstream file(dataFilePath);
        if (!file.is_open()) {
            cerr << "Error opening file: " << dataFilePath << endl;
            return ;
        }

        string line;
        getline(file, line); // Skip the first line

        // Read until the end of the file:
        while (getline(file, line)) {
            // Extract the first entry from the line:
            istringstream iss(line);
            double value;
        
            if (iss >> value) {
                X_test.push_back(value);
                cout << "Push into X_test : " << value << endl; 
            } else {
                cerr << "Error reading value from line: " << line << endl;
            }
        }
         
        Matrix<double> test_tuple_fs(1, 1) ;   
        vector<double> results;
            
        cout << "test_tuple_fs : " << test_tuple_fs << endl;         
        for(unsigned j = 0; j < X_test.size() ; j++){
            test_tuple_fs[0][0] = X_test.at(j);
            cout << "test_tuple_fs, after initialization : " << test_tuple_fs << endl; 
            // Pass the tuple to forward function 
            results.push_back ( net31.forward(test_tuple_fs)[0][0] ); // -- >> Change here 
        }

        cout << "--- We have the results for input file : " << sampleFileName << endl; 

        string modelName = "model_03";  // -- >> Change here 

        string resultFilePath = "results/onnx/" + modelName + "_sample_0" + to_string(i) + ".txt"; 

        // Write results vector to file
        ofstream resultFile(resultFilePath);
        if (!resultFile.is_open()) {
            std::cerr << "Error opening file: " << resultFilePath << std::endl;
            return ; // Handle error appropriately if needed
        }
        resultFile << setprecision(16);
        // Write each result to a separate line
        for(unsigned j = 0; j < results.size(); j++){
            resultFile << results[j] << endl; 
        }
        

        resultFile.close();


        X_test.clear(); 
        file.close();

    }

    //print("-------------------------\n"); 
    cout << "Generating results for model_04 " << endl; 

    for(int i = 1; i <= 5; i++)  {
        dataFolder = "data" ; 
        sampleFileName = "samples_model_0" + to_string(i) ;   
        dataFilePath = dataFolder + "/" + sampleFileName + ".csv" ; 

        cout << "Reading from file : " << dataFilePath << endl; 
        // Read from the second line of the file; 
        // There are two entries per line, read the first  entry only 
        // save the data in double type vector 
   
        // Read from the second line of the file:
        ifstream file(dataFilePath);
        if (!file.is_open()) {
            cerr << "Error opening file: " << dataFilePath << endl;
            return ;
        }

        string line;
        getline(file, line); // Skip the first line

        // Read until the end of the file:
        while (getline(file, line)) {
            // Extract the first entry from the line:
            istringstream iss(line);
            double value;
        
            if (iss >> value) {
                X_test.push_back(value);
                cout << "Push into X_test : " << value << endl; 
            } else {
                cerr << "Error reading value from line: " << line << endl;
            }
        }
         
        Matrix<double> test_tuple_fs(1, 1) ;   
        vector<double> results;
            
        cout << "test_tuple_fs : " << test_tuple_fs << endl;         
        for(unsigned j = 0; j < X_test.size() ; j++){
            test_tuple_fs[0][0] = X_test.at(j);
            cout << "test_tuple_fs, after initialization : " << test_tuple_fs << endl; 
            // Pass the tuple to forward function 
            results.push_back ( net41.forward(test_tuple_fs)[0][0] ); // -- >> Change here 
        }

        cout << "--- We have the results for input file : " << sampleFileName << endl; 

        string modelName = "model_04";  // -- >> Change here 

        string resultFilePath = "results/onnx/" + modelName + "_sample_0" + to_string(i) + ".txt"; 

        // Write results vector to file
        ofstream resultFile(resultFilePath);
        if (!resultFile.is_open()) {
            std::cerr << "Error opening file: " << resultFilePath << std::endl;
            return ; // Handle error appropriately if needed
        }
        resultFile << setprecision(16);
        // Write each result to a separate line
        for(unsigned j = 0; j < results.size(); j++){
            resultFile << results[j] << endl; 
        }
        
        resultFile.close();

        X_test.clear(); 
        file.close();

    }

    //print("-------------------------\n"); 
    cout << "Generating results for model_03 " << endl; 

    for(int i = 1; i <= 5; i++)  {
        dataFolder = "data" ; 
        sampleFileName = "samples_model_0" + to_string(i) ;   
        dataFilePath = dataFolder + "/" + sampleFileName + ".csv" ; 

        cout << "Reading from file : " << dataFilePath << endl; 
        // Read from the second line of the file; 
        // There are two entries per line, read the first  entry only 
        // save the data in double type vector 
   
        // Read from the second line of the file:
        ifstream file(dataFilePath);
        if (!file.is_open()) {
            cerr << "Error opening file: " << dataFilePath << endl;
            return ;
        }

        string line;
        getline(file, line); // Skip the first line

        // Read until the end of the file:
        while (getline(file, line)) {
            // Extract the first entry from the line:
            istringstream iss(line);
            double value;
        
            if (iss >> value) {
                X_test.push_back(value);
                cout << "Push into X_test : " << value << endl; 
            } else {
                cerr << "Error reading value from line: " << line << endl;
            }
        }
         
        Matrix<double> test_tuple_fs(1, 1) ;   
        vector<double> results;
            
        cout << "test_tuple_fs : " << test_tuple_fs << endl;         
        for(unsigned j = 0; j < X_test.size() ; j++){
            test_tuple_fs[0][0] = X_test.at(j);
            cout << "test_tuple_fs, after initialization : " << test_tuple_fs << endl; 
            // Pass the tuple to forward function 
            results.push_back ( net51.forward(test_tuple_fs)[0][0] ); // -- >> Change here 
        }

        cout << "--- We have the results for input file : " << sampleFileName << endl; 

        string modelName = "model_05";  // -- >> Change here 

        string resultFilePath = "results/onnx/" + modelName + "_sample_0" + to_string(i) + ".txt"; 

        // Write results vector to file
        ofstream resultFile(resultFilePath);
        if (!resultFile.is_open()) {
            std::cerr << "Error opening file: " << resultFilePath << std::endl;
            return ; // Handle error appropriately if needed
        }
        resultFile << setprecision(16);
        // Write each result to a separate line
        for(unsigned j = 0; j < results.size(); j++){
            resultFile << results[j] << endl; 
        }
        resultFile.close();

        X_test.clear(); 
        file.close();

    }
}

int main(){

    testModelsOnSampledData(); 

    return 0; 
}