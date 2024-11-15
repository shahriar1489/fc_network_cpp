#!/bin/bash

# Accept the ONNX file name as the first argument                                                          
onnx_file=$1

echo "Processing ONNX file: $onnx_file on $(date '+%A %W %Y %X')"

# Call the Python script and pass the ONNX file to it                                                      

#conda env list 
#conda activate base  # Uncomment if using a conda environment
#python parse_onnx.py $onnx_file
#python python_parser.py $onnx_file
python gemm_parser_v1.py $onnx_file
