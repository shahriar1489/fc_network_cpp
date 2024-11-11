#!/
#!/usr/bin/python
import onnx
from onnx import numpy_helper
import os, sys
import torch 

import numpy as np 
from onnx.reference import ReferenceEvaluator 
from onnx.checker import check_model

from onnx import shape_inference


def traverse_graph(graph):
    for node in graph.node:
        yield node
        
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from traverse_graph(attr.g)

def string_to_dict(node_string):
    
    # Split the string into lines
    raw_lines = node_string.split('\n')
    
    # Process each line
    lines = []
    for line in raw_lines:
        # Strip whitespace from the line
        stripped_line = line.strip()
    
        # If the line is not empty, add it to the list
        if stripped_line:
            lines.append(stripped_line)
    
    node_dict = {}
    
    for line in lines:
        print("--- line\t", line)
        
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip().strip('"')  # Remove quotes
        
        if key in node_dict:
            if isinstance(node_dict[key], list):
                node_dict[key].append(value)
            else:
                node_dict[key] = [node_dict[key], value]
        else:
            node_dict[key] = value
    
    return node_dict

def separate_weights_and_biases(onnx_model_name, onnx_model_path):
    
    print("Inside parsing function")
    # This dictionary below complies with onnx format 
    activations = { 
        "Identity": 0, # this one is not in onnx node proto 
        "Relu": 1,
        "LeakyRelu": 2,
        "Elu": 3,
        "Selu": 4,
        "Sigmoid": 5,
        "HardSigmoid": 6,
        "Tanh": 7,
        "Softplus": 8,
        "Softsign": 9,
        "Celu": 10,
        "Shrink": 11,
        "ThresholdedRelu": 12,
        "Softmax": 13,
        "LogSoftmax": 14,
        "HardSwish": 15,
        "Mish": 16
    }
            
    #activation_names = list(activations.keys())
    activation_codes = []
    noBias = False 
    
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    

    model_name = onnx_model_name 
    folder_dir = "weights_and_biases"

    # Extract the weight tensors and bias tensors from the model
    tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
   
    weights_dict = {}
    biases_dict = {}

    #print("---- how tensor.items() look -----")
    
    #print(tensors.items())
    #print("tensors.items len : ", len(tensors.items()))
    
    #dict_items([('W1', array([[0.1, 0.2],
    #   [0.3, 0.4],
    #   [0.5, 0.6]], dtype=float32)), ('W2', array([[0.7, 0.8],
    #   [0.9, 1. ]], dtype=float32)), ('W3', array([[0.5],
    #   [0.6]], dtype=float32)), ('B1', array([0.1, 0.2], dtype=float32)), ('B2', array([0.3, 0.4], dtype=float32)), ('B3', array([0.5], dtype=float32))])
    
    #print("--------------------------------------")

    # Categorize tensors into weights and biases
    
    
    
    
    for name, tensor in tensors.items():
        """
        print("---------------- printing name and tensor while iterating across tensors.items() ")
        
        print("name : " )
        print(name)
        
        print("tensor : ")
        print(tensor)
        print("tensor type : ", type(tensor)) # numpy array 
    
        
        print("=================")
        """
        if "weight" in name:  # Assuming weight tensors have names starting with 'W'
            weights_dict[name] = tensor  # Add to weights dictionary
            
        
        elif "bias" in name:  # Assuming bias tensors have names starting with 'b'
            biases_dict[name] = tensor  # Add to biases dictionary

        else:  
            print("not weight nor bias.\n") # never comes here 
    
    #print("Print the bias and weights after storing them")
    #print(weights_dict)
    #print(biases_dict)
    print("-------------------------")      
    
    # if biases_dict is empty, then populate it manually ; 
    if(biases_dict == {}): 
        # for each weight tensor detected, create a bias 
        for name, value in weights_dict:
            layer_name = name.split(".")[0] # this convention is ok as long as the nn is made in pytorch 
            
            bias_layer_name = layer_name + ".bias"
            
            # get the row shape of the weight 
            row = value.shape[0]
            
            biases_dict[ bias_layer_name ] = np.zeros(row)
        
    print("--------------inside model.graph.node for " , model_name,  "-----------------")

    
    # activation function is detected in the iteration below 
    for node in model.graph.node:
          
        #print("------------ in node ------ ")
        #print( "\tnode:\n", node )
        print("***")
        #print("node input: ", node.input)
        #print("node output: ", node.output)
        print("node op_type: ", node.op_type)
        print("*** ")
        #print("node type: ", type(node))
        
        #node_input = node.input 
        #node_output = node.output 
        #node_op_type = node.op_type 
                
        # Two things to check below - 
        
        #i) if bias is passed
        
        if( node.op_type =="Gemm"  and  len(node.input) == 3 ): 
            # the two parameters are usually the weight and the output from previous layer      
            print("\t\t---- Gemm w bias layer ---- ")
       
         
        if( node.op_type =="Gemm"  and  len(node.input) == 2 ): 
            # the two parameters are usually the weight and the output from previous layer      
            print("\t\t---- Gemm w/o bias layer ---- ")
        
        #ii) if op_type is activation 
        if( node.op_type in activations  ): 
            print("\t\t---- Activation Code detected: ", node.op_type)
          
            # get the corresponding integer 
            activation_code = activations[ node.op_type  ] 
            #print('-----activation code: ', activation_code)            
            activation_codes.append(activation_code)
          
        
        #print()
        
        #node_str = str(node) 
        #print('----')
        ##print(node_str)
        #print('====\n')
        
        
        #print('--- converting node_str to node_dict ---')
        
                
        #node_dict = string_to_dict(node_str)
        #print(node_dict)

        # check if it's a activation function 
        #if node_dict['op_type'] in activation_names:
            # get the corresponding integer 
            #activation_code = activations[ node_dict['op_type']  ] 

            #print('-----activation code: ', activation_code)            
            #activation_codes.append(activation_code)
            
        print("------------------------------------------")
    
    """
    print("Print the weights and biases...") 
    
    print("Weights:")
    for name, weight in weights_dict.items():
        print(f"Tensor Name: {name}, Tensor Shape: {weight.shape}")

    print("\nBiases:")
    for name, bias in biases_dict.items():
        print(f"Tensor Name: {name}, Tensor Shape: {bias.shape}")

    print("\nActivations: ", activation_codes)
    #for a in activation_codes: 
    #    print(f"Activation code: {a}")
    

    print("model name: ", model_name)
    print('------------------------------')
    """

    # An edge case is not having any activation in the last layer; for classification, 
    

    print("\t\t------- WRITING ON FILE BELOW --------\n\n")
    
    #sys.exit(0)
    
    print("model_name : ", model_name)
    file_name = model_name.split(".")[0] + "_weights_and_biases.txt" # .onnx not needed in naming model
    file_path = os.path.join(folder_dir, file_name)

    # If the file in the path does not exist, create the txt file w/ aforementioned name 
    lengths_match = (len(weights_dict) == len(biases_dict) and len(activation_codes) == len(biases_dict) )   
    
    
    #index_no = 0 # to count the line number 
    
    
    if not lengths_match: # This case is entered for regression problem mostly where the last layer is not activation function
    
        print("\t\t*** Lengths do not match ***")
        print("weights length : ",  len(weights_dict) )
        print("biases length : ",  len(biases_dict))
        

    
        # add identity matrix 
        if( len(activation_codes)  < len(weights_dict) ): 
            diff = len(weights_dict) - len(activation_codes)
            
            for _ in range(diff): 
                activation_codes.append(0)
        print("activation length : " , len(activation_codes))   
        print("***** added Identity function to activation layer")

    #if lengths_match: 
        
    print("\t\t*** Lengths match after modification")
        
        
        # 1. Write on file - in the file path directory, create a txt file 
        # 2. Open the .txt file 
        # 3. On one line weight shape (a, b), and below it 
        # 4. Write the weights , and below it 
        # 5. Write the bias shape, and below it 
        # 6. Write the bias value, and finally 
        # 7. Write the activation code : this is a number  
        #print('--- how about here???')
        
        #print(file_path)
        #sys.exit(0)
        # Write to the txt file
        
    # Set print options to avoid truncating the array output
    np.set_printoptions(threshold=np.inf)
        
    
    print("----- writing weights and activations ----")
        
    with open(file_path, 'w') as f:
        for (w_name, weight), (b_name, bias), activation in zip(weights_dict.items(), biases_dict.items(), activation_codes):
                
            # Issue: for large, multidimensional numpy array, I am getting .... written in file 
                    
            f.write(f"Weight Shape: {weight.shape}\n")
            np.savetxt(f, weight, fmt="%.16f")  # Use 16 decimal places for double precision
            
            f.write(f"Bias Shape: {bias.shape}\n")
            np.savetxt(f, bias, fmt="%.16f")  # Use 16 decimal places for double precision
            
            f.write(f"Activation Code: {activation}\n\n")
                    
            #f.write(f"Weight Shape: {weight.shape}\n{weight}\n")
            #f.write(f"Bias Shape: {bias.shape}\n{bias}\n")
            #f.write(f"Activation Code: {activation}\n\n") 
            
    print(f"Weights, biases, and activation codes successfully written to {file_path}.")

        #print("lengths match!")
    
    #print("------")
    #print()


if __name__ == "__main__":
     
    #onnx_model_path = sys.argv[1] #test() #sys.argv[1]
    
    onnx_model_dir = os.path.join("model", "onnx")
    
    onnx_model = sys.argv[1] #test() #sys.argv[1]
    #onnx_model = "digit.onnx"
    
    onnx_model_path = os.path.join(onnx_model_dir, onnx_model)
    
    print("onnx_model : ", onnx_model)
    print("onnx_model_path", onnx_model_path)
    
    model = onnx.load(onnx_model_path)
    
    print("--- Parser file passed to function")
    separate_weights_and_biases(onnx_model, onnx_model_path)
    #print("-----------------------------------") 
    
    print("\n\n=============")
