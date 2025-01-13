## How to Run the Project

### 1. Setup the Project
#### - Clone the repository from the [GitHub link](https://github.com/shahriar1489/fc_network_cpp/tree/main/fc_network).
#### - Organize your project directory as follows:
  
#### project_root/
#### ├── model/
#### ├── ONNX/        # Place ONNX models here
#### ├── pth/         # PyTorch models (for comparison)
#### ├── weights_and_biases/
#### ├── results/
#### ├── data/
#### ├── src/             # Contains source code files (`layer.cpp`, `fc_network.cpp`, etc.)
#### ├── scripts/         # Python scripts like `gemm_parser_v1.py`


### 2. Compile the Project
#### - Ensure the **Flow*** library is installed and properly linked.
#### - Use the `Makefile` in the project root directory to compile the code:
```bash
make
