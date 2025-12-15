# DFloat11_MPS
DFloat11 is a lossless compression framework introduced in this [paper](https://arxiv.org/abs/2504.11651) that reduces the size of Large Language Models (LLMs) and diffusion models (e.g. FLUX.1, Qwen-Image, etc.) by approximately 30% while preserving bit-for-bit identical outputs to the original model. It enables efficient GPU inference on resource-constrained hardware without sacrificing any accuracy. The [original work](https://github.com/LeanModels/DFloat11) is available for NVIDIA GPUs. The code in this repo is an attempt to port DFloat11 to Apple Silicon.  
## Build Instructions
#### Prerequisites
Clone this repo, create a Python virtual environment, install the requirements specified in the ```requirementes.txt``` file and then verify the following:  
```bash
# Ensure you have Xcode Command Line Tools
xcode-select --install

# Verify the Swift compiler
swiftc --version
```  
#### Using the provided Makefile
```bash
cd dfloat11mps

# Clean build artifacts
make clean

# Build the extension
make

# Test it
make test
```  
## Inference
Just run the ```inference.py``` script, passing a model id from the Hugging Face Hub, the max token count, the temperature and a prompt.  