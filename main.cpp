#include<iostream> 
#include"fc_network.h"
#include<string>

void testOnLargeNetwork(); 
void testModelsOnSampledData(); 

int main(){
    //testModelsOnSampledData(); 
    testOnLargeNetwork(); 
    testModelsOnSampledData(); 

    return 0; 
}

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

void testOnLargeNetwork(){
    
    string reg_onnx = "regression_model.onnx"; 
    FCNetwork reg(reg_onnx);
    
    // At this point, we can successfully read the model

    string filePath = "data/regression_test.txt"; // Update with your actual path
    ifstream file(filePath);
    vector<double> data;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filePath << endl;
        return ;
    }
    
    Matrix<double> m(3, 1); 
    vector< Matrix<double> > results; 
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string value;
        
        // Read each comma-separated value as double
        while (getline(iss, value, ',')) {
            data.push_back(stod(value)); // Convert string to double and add to vector
        }

        cout << "Size of vector : " << data.size() << endl; 

        // Convert the vector to a matrix double and feed it to nn
        for(unsigned i = 0; i < data.size()-1; i++){
            m[i][0] = data.at(i); 
        } // I got one tuple at this point 
        data.clear(); 
        //cout << reg.forward(m); // I got the result I wan  
        results.push_back(reg.forward(m)) ;
        //results.push_back(reg.forward(m)); 

        
        //exit(0); 
    }
    
    file.close();

    //cout << "results size : " , results.size(); 
    //for(unsigned i = 0; i < results.size(); i++){
    //    cout << results.at(i) << endl; 
    //}
    
    // Save the results to a file
    string resultFilePath = "results/onnx/regression_test.txt";
    ofstream resultFile(resultFilePath);
    if (!resultFile.is_open()) {
        cerr << "Error opening file: " << resultFilePath << endl;
        return;
    }
    
    resultFile << setprecision(16);  // Set precision to 16

    // Write each result to the file
    for(unsigned i = 0; i < results.size(); i++) {
        for (unsigned j = 0; j < results.at(i).rows(); j++) {
            for (unsigned k = 0; k < results.at(i).cols(); k++) {
                resultFile << results.at(i)[j][k];
                if (k < results.at(i).cols() - 1) resultFile << ", "; // Comma-separated for columns
            }
            resultFile << endl; // New line for each row
        }
    }

    resultFile.close();

    cout << "Results saved to " << resultFilePath << endl;
}