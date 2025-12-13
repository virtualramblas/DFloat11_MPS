# DFloat11_MPS
DFloat11 for Apple Silicon.  
## Build Instructions
### Prerequisites
```bash
# Ensure you have Xcode Command Line Tools
xcode-select --install

# Verify the Swift compiler
swiftc --version
```  
### Using the provide Makefile
```bash
cd dfloat11mps

# Build the extension
make

# Test it
make test

# Clean build artifacts
make clean
```  
