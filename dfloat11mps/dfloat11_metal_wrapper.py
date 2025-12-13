"""
Python wrapper for the Swift Metal decoder.
This uses ctypes to call the C-compatible functions in the Swift library.
"""

import ctypes
import numpy as np
import torch
from pathlib import Path
import sys

class MetalDecoder:
    """Python wrapper for Swift Metal decoder."""
    
    def __init__(self):
        # Load the Swift library
        lib_path = self._find_library()
        if lib_path is None:
            raise RuntimeError("Could not find libdfloat11metal.dylib. Please run: python setup.py build_ext --inplace")
        
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define function signatures
        self.lib.dfloat11_create_decoder.restype = ctypes.c_void_p
        self.lib.dfloat11_create_decoder.argtypes = []
        
        self.lib.dfloat11_release_decoder.restype = None
        self.lib.dfloat11_release_decoder.argtypes = [ctypes.c_void_p]
        
        self.lib.dfloat11_decode.restype = ctypes.c_int32
        self.lib.dfloat11_decode.argtypes = [
            ctypes.c_void_p,  # decoder
            ctypes.c_void_p,  # luts
            ctypes.c_size_t,  # luts_size (changed from c_int)
            ctypes.c_void_p,  # encoded
            ctypes.c_size_t,  # encoded_size (changed from c_int)
            ctypes.c_void_p,  # sign_mantissa
            ctypes.c_size_t,  # sign_size (changed from c_int)
            ctypes.c_void_p,  # output_positions
            ctypes.c_size_t,  # outpos_size (changed from c_int)
            ctypes.c_void_p,  # gaps
            ctypes.c_size_t,  # gaps_size (changed from c_int)
            ctypes.c_void_p,  # output
            ctypes.c_size_t,  # output_size (changed from c_int)
            ctypes.c_uint32,  # n_luts
            ctypes.c_uint32,  # n_bytes
            ctypes.c_uint32,  # n_elements
            ctypes.c_size_t,  # shared_mem_size (changed from c_int)
            ctypes.c_size_t,  # threads_per_block (changed from c_int)
        ]
        
        # Create decoder instance
        self.decoder = self.lib.dfloat11_create_decoder()
        if not self.decoder:
            raise RuntimeError("Failed to create Metal decoder")
        
        print("Metal decoder initialized successfully via Swift", file=sys.stderr)
    
    def _find_library(self):
        """Find the compiled Swift library."""
        # Check current directory
        candidates = [
            Path(__file__).parent / "libdfloat11metal.dylib",
            Path("libdfloat11metal.dylib"),
            Path("build") / "libdfloat11metal.dylib",
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        return None
    
    def decode(self, luts_tensor, encoded_tensor, sign_tensor, 
               outpos_tensor, gaps_tensor, output_tensor,
               n_luts, n_bytes, n_elements, shared_mem_size, threads_per_block):
        """
        Execute Metal decode kernel.
        
        Args:
            All tensors should be on CPU (numpy arrays or CPU torch tensors)
            The output will be written to output_tensor
        """
        # Convert PyTorch tensors to numpy if needed
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.cpu().numpy()
            return t
        
        luts_np = to_numpy(luts_tensor)
        encoded_np = to_numpy(encoded_tensor)
        sign_np = to_numpy(sign_tensor)
        outpos_np = to_numpy(outpos_tensor)
        gaps_np = to_numpy(gaps_tensor)
        output_np = to_numpy(output_tensor)
        
        # Ensure contiguous arrays
        luts_np = np.ascontiguousarray(luts_np)
        encoded_np = np.ascontiguousarray(encoded_np)
        sign_np = np.ascontiguousarray(sign_np)
        outpos_np = np.ascontiguousarray(outpos_np)
        gaps_np = np.ascontiguousarray(gaps_np)
        output_np = np.ascontiguousarray(output_np)
        
        # Get pointers
        luts_ptr = luts_np.ctypes.data_as(ctypes.c_void_p)
        encoded_ptr = encoded_np.ctypes.data_as(ctypes.c_void_p)
        sign_ptr = sign_np.ctypes.data_as(ctypes.c_void_p)
        outpos_ptr = outpos_np.ctypes.data_as(ctypes.c_void_p)
        gaps_ptr = gaps_np.ctypes.data_as(ctypes.c_void_p)
        output_ptr = output_np.ctypes.data_as(ctypes.c_void_p)
        
        # Call Swift function
        result = self.lib.dfloat11_decode(
            self.decoder,
            luts_ptr, ctypes.c_size_t(luts_np.nbytes),
            encoded_ptr, ctypes.c_size_t(encoded_np.nbytes),
            sign_ptr, ctypes.c_size_t(sign_np.nbytes),
            outpos_ptr, ctypes.c_size_t(outpos_np.nbytes),
            gaps_ptr, ctypes.c_size_t(gaps_np.nbytes),
            output_ptr, ctypes.c_size_t(output_np.nbytes),
            ctypes.c_uint32(n_luts),
            ctypes.c_uint32(n_bytes),
            ctypes.c_uint32(n_elements),
            ctypes.c_size_t(shared_mem_size),
            ctypes.c_size_t(threads_per_block)
        )
        
        if result != 0:
            raise RuntimeError(f"Metal decode failed with error code: {result}")
        
        # Copy result back to output tensor if it's a PyTorch tensor
        if isinstance(output_tensor, torch.Tensor):
            output_tensor.copy_(torch.from_numpy(output_np))
    
    def __del__(self):
        """Clean up decoder instance."""
        if hasattr(self, 'decoder') and self.decoder:
            self.lib.dfloat11_release_decoder(self.decoder)


# Singleton instance
_metal_decoder = None


def get_metal_decoder():
    """Get or create the Metal decoder singleton."""
    global _metal_decoder
    if _metal_decoder is None:
        _metal_decoder = MetalDecoder()
    return _metal_decoder

def get_hook_swift(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook using the Swift Metal decoder.
    """
    if isinstance(threads_per_block, (list, tuple)):
        threads_per_block = threads_per_block[0]

    def decode_hook(module, _):
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]

        device = module.encoded_exponent.device
        
        # Move tensors to CPU for Swift processing
        luts_cpu = module.luts.cpu()
        encoded_cpu = module.encoded_exponent.cpu()
        sign_cpu = module.sign_mantissa.cpu()
        outpos_cpu = module.output_positions.cpu()
        gaps_cpu = module.gaps.cpu()
        
        # Create output tensor on CPU
        output_cpu = torch.empty(n_elements, dtype=torch.uint16)
        
        # Get Metal decoder and execute
        decoder = get_metal_decoder()
        decoder.decode(
            luts_tensor=luts_cpu,
            encoded_tensor=encoded_cpu,
            sign_tensor=sign_cpu,
            outpos_tensor=outpos_cpu,
            gaps_tensor=gaps_cpu,
            output_tensor=output_cpu,
            n_luts=n_luts,
            n_bytes=n_bytes,
            n_elements=n_elements,
            shared_mem_size=module.shared_mem_size,
            threads_per_block=threads_per_block
        )
        
        # Convert output to correct dtype and move to target device
        if device.type == 'mps':
            # View as float16 or bfloat16 and move to MPS
            if hasattr(module, 'weight_dtype'):
                dtype = module.weight_dtype
            else:
                dtype = torch.float16  # Default
            
            reconstructed = output_cpu.view(dtype).to(device)
        else:
            reconstructed = output_cpu.view(torch.float16).to(device)

        # Inject reconstructed weights
        if isinstance(module, torch.nn.Linear):
            module.weight = reconstructed.view(module.out_features, module.in_features)
        elif isinstance(module, torch.nn.Embedding):
            module.weight = reconstructed.view(module.num_embeddings, module.embedding_dim)
        else:
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                sub_module.weight = weight.view(sub_module.out_features, sub_module.in_features)

    return decode_hook