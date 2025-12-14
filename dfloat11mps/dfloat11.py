import math
import os
import re
import platform
from sys import stderr
from typing import Optional, Dict, Union
from tqdm import tqdm

import torch
import torch.nn as nn

# Metal support for Apple Silicon
from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLResourceStorageModeShared,
)

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
from transformers.modeling_utils import no_init_weights
# Check if we're on Apple Silicon
USE_SWIFT_METAL = (
    platform.system() == 'Darwin' and 
    platform.machine() == 'arm64'
)
if USE_SWIFT_METAL:
    from dfloat11mps.dfloat11_metal_wrapper import get_metal_decoder, get_hook_swift
else:
    pass

# Metal kernel source code
METAL_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void dfloat11_decode(
    device const uint8_t* lookup_table [[buffer(0)]],
    device const uint8_t* compressed_data [[buffer(1)]],
    device const uint8_t* sign_mantissa [[buffer(2)]],
    device uint32_t* output_positions [[buffer(3)]],
    device const uint8_t* gaps [[buffer(4)]],
    device uint16_t* output [[buffer(5)]],
    constant uint32_t& n_luts [[buffer(6)]],
    constant uint32_t& n_bytes [[buffer(7)]],
    constant uint32_t& n_elements [[buffer(8)]],
    threadgroup uint32_t* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    const uint LOOKUP_OFFSET = 256;
    uint global_id = bid * threads_per_group + tid;
    uint input_byte_offset = global_id * 8;
    
    if (input_byte_offset >= n_bytes) {
        return;
    }
    
    // Load 8 bytes of compressed data into 64-bit buffer
    uint64_t buffer = 0;
    for (uint i = 0; i < 8 && (input_byte_offset + i) < n_bytes; i++) {
        buffer |= uint64_t(compressed_data[input_byte_offset + i]) << (56 - i * 8);
    }
    
    // Load 4 bytes lookahead
    uint32_t lookahead = 0;
    for (uint i = 0; i < 4 && (input_byte_offset + 8 + i) < n_bytes; i++) {
        lookahead |= uint32_t(compressed_data[input_byte_offset + 8 + i]) << (24 - i * 8);
    }
    
    // Decode initial code from gaps table
    uint32_t gap_offset = global_id * 5;
    uint32_t byte_idx = gap_offset / 8;
    uint32_t bit_offset = gap_offset % 8;
    
    uint8_t initial_bits = 0;
    if (byte_idx + 1 < n_bytes) {
        uint16_t two_bytes = (uint16_t(gaps[byte_idx]) << 8) | uint16_t(gaps[byte_idx + 1]);
        initial_bits = uint8_t((two_bytes >> (11 - bit_offset)) & 0x1F);
    }
    
    buffer <<= (initial_bits & 0x1F);
    uint8_t bits_consumed = initial_bits & 0x1F;
    
    // Phase 1: Decode until we have 32+ bits
    uint32_t sample_count = 0;
    
    while (bits_consumed < 32) {
        uint8_t lookup_idx = uint8_t(buffer >> 56);
        uint8_t code_byte = lookup_table[lookup_idx];
        
        // Multi-level lookup for long codes
        while ((code_byte & 0xFF) >= 240) {
            uint16_t offset = (uint16_t(code_byte) << 8) & 0xFF00;
            uint8_t shift = (code_byte >= 243) ? 32 : (code_byte >= 241) ? 40 : 48;
            uint8_t next_idx = uint8_t((buffer >> shift) & 0xFF);
            code_byte = lookup_table[offset | next_idx];
        }
        
        uint8_t bit_length = lookup_table[LOOKUP_OFFSET + (code_byte & 0xFF)];
        buffer <<= bit_length;
        bits_consumed += bit_length;
        sample_count++;
    }
    
    // Combine buffer with lookahead
    bits_consumed -= 32;
    uint64_t combined = (buffer >> bits_consumed) | (uint64_t(lookahead) << (32 - bits_consumed));
    
    // Phase 2: Continue decoding
    while (((bits_consumed & 0xF8) + 4) <= 64) {
        uint8_t lookup_idx = uint8_t(combined >> 56);
        uint8_t code_byte = lookup_table[lookup_idx];
        
        while ((code_byte & 0xFF) >= 240) {
            uint16_t offset = (uint16_t(code_byte) << 8) & 0xFF00;
            uint8_t shift = (code_byte >= 243) ? 32 : (code_byte >= 241) ? 40 : 48;
            uint8_t next_idx = uint8_t((combined >> shift) & 0xFF);
            code_byte = lookup_table[offset | next_idx];
        }
        
        uint8_t bit_length = lookup_table[LOOKUP_OFFSET + (code_byte & 0xFF)];
        combined <<= bit_length;
        bits_consumed += bit_length;
        sample_count++;
    }
    
    // Store sample count in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared_mem[tid * 4] = sample_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel prefix sum (up-sweep)
    for (uint stride = 2; stride <= threads_per_group; stride *= 2) {
        if (((tid + 1) & (stride - 1)) == 0) {
            uint left_idx = (tid - stride / 2) * 4;
            shared_mem[tid * 4] += shared_mem[left_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Down-sweep
    if (tid == 0) {
        shared_mem[(threads_per_group - 1) * 4 + 4] = shared_mem[(threads_per_group - 1) * 4];
        shared_mem[(threads_per_group - 1) * 4] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = threads_per_group; stride >= 2; stride /= 2) {
        if (((tid + 1) & (stride - 1)) == 0) {
            uint left_idx = (tid - stride / 2) * 4;
            uint temp = shared_mem[left_idx];
            shared_mem[left_idx] = shared_mem[tid * 4];
            shared_mem[tid * 4] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Get output offset
    uint32_t output_offset = shared_mem[tid * 4];
    uint32_t block_start = (bid > 0) ? output_positions[bid - 1] : 0;
    uint32_t write_base = block_start + output_offset;
    
    // Decode and write samples
    for (uint32_t i = 0; i < sample_count && (write_base + i) < n_elements; i++) {
        uint8_t lookup_idx = uint8_t(combined >> 56);
        uint8_t code_byte = lookup_table[lookup_idx];
        
        while ((code_byte & 0xFF) >= 240) {
            uint16_t offset = (uint16_t(code_byte) << 8) & 0xFF00;
            uint8_t shift = (code_byte >= 243) ? 32 : (code_byte >= 241) ? 40 : 48;
            uint8_t next_idx = uint8_t((combined >> shift) & 0xFF);
            code_byte = lookup_table[offset | next_idx];
        }
        
        // Extract and combine sign and mantissa
        uint8_t sign_byte = sign_mantissa[write_base + i];
        uint16_t result = ((sign_byte & 0x80) << 8) |
                         (((sign_byte & 0x7F) << 9) |
                         (((code_byte & 0xFE) >> 1) << 7) |
                         (code_byte & 0x01));
        
        output[write_base + i] = result;
        
        uint8_t bit_length = lookup_table[LOOKUP_OFFSET + (code_byte & 0xFF)];
        combined <<= bit_length;
    }
    
    // Update output positions
    if (tid == threads_per_group - 1) {
        output_positions[bid] = block_start + shared_mem[tid * 4 + 4];
    }
}
"""


class MetalDecoder:
    """Metal-based decoder for Apple Silicon."""
    
    def __init__(self):
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not supported on this device")
        
        print(f"Using Metal device: {self.device.name()}", file=stderr)
        
        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")
        
        # Compile Metal kernel
        # newLibraryWithSource_options_error_ returns (library, error) tuple
        library_result = self.device.newLibraryWithSource_options_error_(
            METAL_KERNEL_SOURCE, None, None
        )
        
        # Unpack the tuple
        if isinstance(library_result, tuple):
            self.library, error = library_result
        else:
            self.library = library_result
            error = None
        
        if self.library is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to compile Metal kernel: {error_msg}")
        
        self.function = self.library.newFunctionWithName_("dfloat11_decode")
        if self.function is None:
            raise RuntimeError("Failed to find dfloat11_decode function in Metal library")
        
        # Create compute pipeline state
        # newComputePipelineStateWithFunction_error_ also returns (pipeline, error) tuple
        pipeline_result = self.device.newComputePipelineStateWithFunction_error_(
            self.function, None
        )
        
        # Unpack the tuple
        if isinstance(pipeline_result, tuple):
            self.pipeline, error = pipeline_result
        else:
            self.pipeline = pipeline_result
            error = None
        
        if self.pipeline is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to create Metal compute pipeline: {error_msg}")
        
        print("Metal decoder initialized successfully", file=stderr)
    
    def _create_metal_buffer_from_tensor(self, tensor):
        """Create or get Metal buffer from PyTorch MPS tensor."""
        if tensor.device.type != 'mps':
            raise ValueError(f"Tensor must be on MPS device, got {tensor.device}")
        
        # Get tensor properties
        size_bytes = tensor.numel() * tensor.element_size()
        
        # Move tensor to CPU to get the data
        cpu_tensor = tensor.cpu().contiguous()
        
        # Handle bfloat16 which numpy doesn't support
        if cpu_tensor.dtype == torch.bfloat16:
            # View bfloat16 as uint16 for numpy compatibility
            cpu_tensor = cpu_tensor.view(torch.uint16)
        
        # Convert to numpy array for easier copying
        numpy_data = cpu_tensor.numpy()
        
        # Create Metal buffer with the data
        # Using newBufferWithBytes creates and copies in one step
        buffer = self.device.newBufferWithBytes_length_options_(
            numpy_data.tobytes(),
            size_bytes,
            MTLResourceStorageModeShared
        )
        
        if buffer is None:
            raise RuntimeError(f"Failed to allocate Metal buffer of size {size_bytes}")
        
        return buffer
    
    def decode(self, luts_tensor, encoded_exp_tensor, sign_mant_tensor, 
               out_pos_tensor, gaps_tensor, output_tensor,
               n_luts, n_bytes, n_elements, shared_mem_size, threads_per_block):
        """Execute Metal kernel for decoding."""
        
        try:
            # Create Metal buffers from PyTorch tensors
            print(f"Creating Metal buffers... luts: {luts_tensor.shape}, encoded: {encoded_exp_tensor.shape}", file=stderr)
            luts_buffer = self._create_metal_buffer_from_tensor(luts_tensor)
            encoded_buffer = self._create_metal_buffer_from_tensor(encoded_exp_tensor)
            sign_buffer = self._create_metal_buffer_from_tensor(sign_mant_tensor)
            outpos_buffer = self._create_metal_buffer_from_tensor(out_pos_tensor)
            gaps_buffer = self._create_metal_buffer_from_tensor(gaps_tensor)
            output_buffer = self._create_metal_buffer_from_tensor(output_tensor)
            print("Metal buffers created successfully", file=stderr)
            
            # Create command buffer
            command_buffer = self.command_queue.commandBuffer()
            if command_buffer is None:
                raise RuntimeError("Failed to create Metal command buffer")
            
            # Create compute encoder
            encoder = command_buffer.computeCommandEncoder()
            if encoder is None:
                raise RuntimeError("Failed to create Metal compute encoder")
            
            # Set compute pipeline
            encoder.setComputePipelineState_(self.pipeline)
            
            # Set buffers
            encoder.setBuffer_offset_atIndex_(luts_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(encoded_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(sign_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(outpos_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(gaps_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
            
            # Set constants using struct packing
            import struct
            n_luts_bytes = struct.pack('<I', n_luts)
            n_bytes_bytes = struct.pack('<I', n_bytes)
            n_elements_bytes = struct.pack('<I', n_elements)
            
            encoder.setBytes_length_atIndex_(n_luts_bytes, 4, 6)
            encoder.setBytes_length_atIndex_(n_bytes_bytes, 4, 7)
            encoder.setBytes_length_atIndex_(n_elements_bytes, 4, 8)
            
            # Set threadgroup memory
            print(f"Setting threadgroup memory: {shared_mem_size} bytes", file=stderr)
            encoder.setThreadgroupMemoryLength_atIndex_(shared_mem_size, 0)
            
            # Calculate grid size
            blocks_per_grid = math.ceil(n_bytes / (threads_per_block * 8))
            total_threads = blocks_per_grid * threads_per_block
            
            print(f"Dispatch config: blocks={blocks_per_grid}, threads_per_block={threads_per_block}, total={total_threads}", file=stderr)
            print(f"Kernel params: n_luts={n_luts}, n_bytes={n_bytes}, n_elements={n_elements}", file=stderr)
            
            # Check threadgroup memory limits (if available)
            try:
                max_threadgroup_memory = self.pipeline.maxTotalThreadgroupMemory()
                print(f"Max threadgroup memory: {max_threadgroup_memory} bytes, requested: {shared_mem_size} bytes", file=stderr)
                if shared_mem_size > max_threadgroup_memory:
                    raise RuntimeError(f"Requested threadgroup memory ({shared_mem_size} bytes) exceeds "
                                     f"device limit ({max_threadgroup_memory} bytes)")
            except AttributeError:
                # Method might not be available on all devices, just warn
                print(f"Warning: Could not check threadgroup memory limits (requested: {shared_mem_size} bytes)", file=stderr)
            
            # Check thread limits (if available)
            try:
                max_threads = self.pipeline.maxTotalThreadsPerThreadgroup()
                print(f"Max threads per threadgroup: {max_threads}, requested: {threads_per_block}", file=stderr)
                if threads_per_block > max_threads:
                    raise RuntimeError(f"Requested threads per threadgroup ({threads_per_block}) exceeds "
                                     f"device limit ({max_threads})")
            except AttributeError:
                # Method might not be available on all devices, just warn
                print(f"Warning: Could not check thread limits (requested: {threads_per_block})", file=stderr)
            
            # Create MTLSize objects for dispatch
            from Metal import MTLSizeMake
            grid_size = MTLSizeMake(total_threads, 1, 1)
            threadgroup_size = MTLSizeMake(threads_per_block, 1, 1)
            
            # Dispatch threads
            print("Dispatching Metal kernel...", file=stderr)
            encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
            
            # End encoding
            encoder.endEncoding()
            print("Encoder ended, committing command buffer...", file=stderr)
            
            # Commit and wait
            command_buffer.commit()
            print("Command buffer committed, waiting for completion...", file=stderr)
            command_buffer.waitUntilCompleted()
            print("Command buffer completed", file=stderr)
            
            # Check for errors
            status = command_buffer.status()
            print(f"Command buffer status: {status}", file=stderr)
            
            if status == 5:  # MTLCommandBufferStatusError
                error = command_buffer.error()
                raise RuntimeError(f"Metal command buffer execution failed: {error}")
            
            # Copy output buffer back to PyTorch tensor
            print("Copying output buffer back to PyTorch tensor...", file=stderr)
            output_size_bytes = output_tensor.numel() * output_tensor.element_size()
            print(f"Output buffer size: {output_size_bytes} bytes ({output_tensor.numel()} elements)", file=stderr)
            
            import numpy as np
            from Foundation import NSData
            
            print("Reading Metal buffer data using NSData...", file=stderr)
            
            try:
                # Use NSData to wrap the buffer - this should work with Metal buffers
                # Create NSData from the buffer without copying
                ns_data = NSData.dataWithBytesNoCopy_length_freeWhenDone_(
                    output_buffer.contents(),
                    output_size_bytes,
                    False  # Don't free when done - Metal owns this memory
                )
                
                if ns_data is None:
                    raise RuntimeError("Failed to create NSData from Metal buffer")
                
                print(f"Created NSData wrapper, length: {ns_data.length()}", file=stderr)
                
                # Get bytes from NSData
                # NSData.bytes() returns a pointer we can work with
                ns_bytes = ns_data.bytes()
                
                # Now convert using ctypes or objc methods
                # Try to get the data as a Python bytes object
                try:
                    # Method 1: Use tobytes() if available
                    python_bytes = ns_data.bytes().tobytes(output_size_bytes)
                    print(f"Got bytes using tobytes(), length: {len(python_bytes)}", file=stderr)
                except AttributeError:
                    # Method 2: Create bytes from NSData using list comprehension
                    print("tobytes() not available, reading byte-by-byte from NSData...", file=stderr)
                    byte_list = []
                    for i in range(output_size_bytes):
                        # Access bytes using getBytes:range:
                        byte_val = ns_data.bytes()[i]
                        byte_list.append(byte_val)
                    python_bytes = bytes(byte_list)
                    print(f"Read {len(python_bytes)} bytes", file=stderr)
                
                # Convert to numpy array
                if output_tensor.dtype == torch.float16:
                    numpy_output = np.frombuffer(python_bytes, dtype=np.float16)
                    cpu_output = torch.from_numpy(numpy_output.copy())
                elif output_tensor.dtype == torch.bfloat16:
                    numpy_output = np.frombuffer(python_bytes, dtype=np.uint16)
                    cpu_output = torch.from_numpy(numpy_output.copy()).view(torch.bfloat16)
                else:
                    numpy_output = np.frombuffer(python_bytes, dtype=np.uint16)
                    cpu_output = torch.from_numpy(numpy_output.copy())
                
                print(f"Successfully created numpy array with {len(numpy_output)} elements", file=stderr)
                
            except Exception as e:
                print(f"Error with NSData method: {e}", file=stderr)
                import traceback
                traceback.print_exc()
                
                # Last resort: use getBytes:length: from NSData
                print("Trying NSData getBytes:length: method...", file=stderr)
                try:
                    ns_data = NSData.dataWithBytesNoCopy_length_freeWhenDone_(
                        output_buffer.contents(),
                        output_size_bytes,
                        False
                    )
                    
                    # Create a mutable buffer to receive the bytes
                    import ctypes
                    buffer = (ctypes.c_ubyte * output_size_bytes)()
                    
                    # Copy data from NSData to our buffer
                    ns_data.getBytes_length_(buffer, output_size_bytes)
                    
                    # Convert to bytes
                    python_bytes = bytes(buffer)
                    
                    if output_tensor.dtype == torch.float16:
                        numpy_output = np.frombuffer(python_bytes, dtype=np.float16)
                        cpu_output = torch.from_numpy(numpy_output.copy())
                    elif output_tensor.dtype == torch.bfloat16:
                        numpy_output = np.frombuffer(python_bytes, dtype=np.uint16)
                        cpu_output = torch.from_numpy(numpy_output.copy()).view(torch.bfloat16)
                    else:
                        numpy_output = np.frombuffer(python_bytes, dtype=np.uint16)
                        cpu_output = torch.from_numpy(numpy_output.copy())
                    
                    print(f"Successfully read buffer using getBytes:length:", file=stderr)
                    
                except Exception as e2:
                    print(f"Error with getBytes:length: {e2}", file=stderr)
                    traceback.print_exc()
                    raise RuntimeError(f"Cannot read Metal buffer data. All methods failed. Last error: {e2}")
            
            # Reshape to match output tensor shape
            cpu_output = cpu_output.reshape(output_tensor.shape)
            print(f"Reshaped to {cpu_output.shape}", file=stderr)
            
            # Copy back to original MPS tensor
            print("Copying result back to MPS tensor...", file=stderr)
            output_tensor.copy_(cpu_output)
            print("Metal decode completed successfully", file=stderr)
            
        except Exception as e:
            print(f"Error during Metal decode: {type(e).__name__}: {e}", file=stderr)
            import traceback
            traceback.print_exc()
            raise


# Global Metal decoder instance
_metal_decoder = None


def get_metal_decoder():
    """Get or create Metal decoder singleton."""
    global _metal_decoder
    if _metal_decoder is None:
        _metal_decoder = MetalDecoder()
    return _metal_decoder


class TensorManager:
    """
    Static utility class that manages tensor allocation and reuse
    to minimize memory allocation overhead during tensor reconstruction.
    Automatically selects float16 or bfloat16 based on device capabilities.
    """
    _tensors = {}
    _device_dtype_cache = {}

    @staticmethod
    def _get_optimal_dtype(device):
        """
        Determine the optimal dtype for the given device.
        Returns bfloat16 if supported, otherwise float16.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check cache first
        if device in TensorManager._device_dtype_cache:
            return TensorManager._device_dtype_cache[device]
        
        # Determine optimal dtype based on device
        if device.type == 'mps':
            # Check if MPS supports bfloat16
            # M1/M2 chips (before M3) don't support bfloat16
            try:
                # Try to create a small bfloat16 tensor on MPS
                test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=device)
                optimal_dtype = torch.bfloat16
                print(f"Device {device} supports bfloat16", file=stderr)
            except (RuntimeError, TypeError):
                # bfloat16 not supported, fall back to float16
                optimal_dtype = torch.float16
                print(f"Device {device} does not support bfloat16, using float16", file=stderr)
        elif device.type == 'cuda':
            # Most CUDA devices support bfloat16
            optimal_dtype = torch.bfloat16
        else:
            # CPU and other devices: use float16 as safe default
            optimal_dtype = torch.float16
        
        # Cache the result
        TensorManager._device_dtype_cache[device] = optimal_dtype
        return optimal_dtype

    @staticmethod
    def get_tensor(device, n_elements, dtype=None):
        """
        Get a float16/bfloat16 tensor with at least n_elements on the specified device.
        
        Args:
            device: Target device
            n_elements: Number of elements needed
            dtype: Optional dtype override. If None, automatically selects optimal dtype.
        
        Returns:
            Tensor with at least n_elements on the specified device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Determine dtype
        if dtype is None:
            dtype = TensorManager._get_optimal_dtype(device)
        
        # Check if we have a cached tensor
        cache_key = (device, dtype)
        if cache_key in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[cache_key]
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            # Tensor too small, delete and reallocate
            del TensorManager._tensors[cache_key]
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # Allocate new tensor
        new_tensor = torch.empty(n_elements, dtype=dtype, device=device)
        dtype_name = 'bf16' if dtype == torch.bfloat16 else 'fp16'
        print(f'Allocated {n_elements} {dtype_name} elements on device {device}', file=stderr)
        
        TensorManager._tensors[cache_key] = new_tensor
        return new_tensor

    @staticmethod
    def clear_device(device=None, dtype=None):
        """
        Clear tensors for a specific device or all devices.
        
        Args:
            device: Device to clear, or None to clear all
            dtype: Dtype to clear, or None to clear all dtypes
        """
        if device is None and dtype is None:
            # Clear everything
            TensorManager._tensors.clear()
            TensorManager._device_dtype_cache.clear()
        elif device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            
            # Clear specific device (all dtypes or specific dtype)
            keys_to_delete = []
            for key in TensorManager._tensors.keys():
                cached_device, cached_dtype = key
                if cached_device == device:
                    if dtype is None or cached_dtype == dtype:
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del TensorManager._tensors[key]
            
            # Clear dtype cache for this device
            if device in TensorManager._device_dtype_cache:
                del TensorManager._device_dtype_cache[device]
        
        # Clear MPS cache if applicable
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def get_hook(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights
    using Metal on Apple Silicon.
    """
    if USE_SWIFT_METAL:
        return get_hook_swift(threads_per_block, bytes_per_thread)
    else:
        return None

def load_and_replace_tensors(model, directory_path, dfloat11_config):
    """
    Loads DFloat11 compressed weights from safetensors files
    and configures Metal-based decompression.
    """
    threads_per_block = dfloat11_config['threads_per_block']
    bytes_per_thread  = dfloat11_config['bytes_per_thread']
    pattern_dict      = dfloat11_config['pattern_dict']
    
    safetensors_files = [f for f in os.listdir(directory_path) if f.endswith('.safetensors')]
    for file_name in tqdm(safetensors_files, desc='Loading DFloat11 safetensors'):
        file_path = os.path.join(directory_path, file_name)
        loaded_tensors = load_file(file_path)
        
        for tensor_name, tensor_value in loaded_tensors.items():
            if tensor_name in model.state_dict():
                if tensor_name in dict(model.named_parameters()):
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}", file=stderr)
                else:
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}", file=stderr)
            else:
                parts = tensor_name.split('.')
                module = model
                
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        print(f"Cannot find module path for {tensor_name}", file=stderr)
                        break
                else:
                    if parts[-1] == 'split_positions':
                        setattr(module, 'split_positions', tensor_value.tolist())
                    else:
                        module.register_buffer(parts[-1], tensor_value)

                    if parts[-1] == 'encoded_exponent':
                        module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))

                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, '.'.join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                else:
                                    setattr(module, 'weight_injection_modules', [])
                                    for attr_path in attr_names:
                                        attr_parts = attr_path.split('.')
                                        target = module
                                        for p in attr_parts:
                                            target = getattr(target, p)
                                        tmp = target.weight
                                        delattr(target, 'weight')
                                        del tmp
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == 'output_positions':
                        setattr(
                            module,
                            'shared_mem_size',
                            threads_per_block[0] * 4 + 4 + (
                                module.output_positions.view(torch.uint32).numpy()[1:] - \
                                    module.output_positions.view(torch.uint32).numpy()[:-1]
                            ).max().item() * 2
                        )
    
    return model


def get_no_split_classes(model, pattern_dict):
    """Find model layer classes that should not be split across devices."""
    no_split_classes = []
    for pattern in pattern_dict:
        for full_name, sub_module in model.named_modules():
            if re.fullmatch(pattern, full_name):
                class_name = sub_module.__class__.__name__
                if class_name not in no_split_classes:
                    no_split_classes.append(class_name)
    return no_split_classes


class DFloat11Model:
    """
    Wrapper class for loading models with DFloat11 compressed weights
    on Apple Silicon using Metal Performance Shaders.
    """
    @classmethod
    def from_pretrained(
        cls,
        dfloat11_model_name_or_path: str,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        bfloat16_model = None,
        **kwargs,
    ):
        """
        Load a model with DFloat11 compressed weights for Apple Silicon.
        
        Args:
            dfloat11_model_name_or_path: Local path or HF Hub model name
            device: Target device ('mps' or None for auto-detect)
            device_map: Strategy for distributing model across devices
            max_memory: Maximum memory allocation per device
            bfloat16_model: Optional pre-initialized model to load weights into
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_config
            
        Returns:
            Model with DFloat11 compressed weights configured for Metal decompression
        """
        # Check if MPS is available
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS (Metal Performance Shaders) is not available. "
                             "Please ensure you're running on Apple Silicon with macOS 12.3+")
        
        print("DFloat11 Model for Apple Silicon (Metal backend)", file=stderr)
        
        # Detect optimal dtype for this device
        test_device = torch.device('mps')
        optimal_dtype = TensorManager._get_optimal_dtype(test_device)
        print(f"Using dtype: {optimal_dtype}", file=stderr)
        
        # Resolve model path
        if os.path.exists(dfloat11_model_name_or_path):
            dfloat11_model_path = dfloat11_model_name_or_path
        else:
            dfloat11_model_path = dfloat11_model_name_or_path.replace('/', '__')
            if not os.path.exists(dfloat11_model_path):
                print(f"Downloading model from {dfloat11_model_name_or_path}...", file=stderr)
                snapshot_download(dfloat11_model_name_or_path, local_dir=dfloat11_model_path)

        # Load model configuration
        config = AutoConfig.from_pretrained(dfloat11_model_path)
        
        if bfloat16_model:
            model = bfloat16_model
        else:
            # Use optimal dtype for this device
            model_dtype = optimal_dtype
            
            with no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=model_dtype, **kwargs,
                )
                model.tie_weights()
                model.eval()

            try:
                generation_config = GenerationConfig.from_pretrained(dfloat11_model_path)
                model.generation_config = generation_config
            except Exception:
                pass

        assert hasattr(config, 'dfloat11_config'), \
            "Model configuration must contain 'dfloat11_config'"
        dfloat11_config = config.dfloat11_config

        # Load compressed weights
        load_and_replace_tensors(model, dfloat11_model_path, dfloat11_config)

        # Calculate model size
        model_bytes = 0
        for param in model.state_dict().values():
            if param.dtype in [torch.uint8, torch.int8]:
                model_bytes += param.numel()
            elif param.dtype in [torch.float16, torch.bfloat16, torch.int16, torch.uint16]:
                model_bytes += param.numel() * 2
            elif param.dtype in [torch.float32, torch.int32, torch.uint32]:
                model_bytes += param.numel() * 4
            elif param.dtype in [torch.float64, torch.int64, torch.uint64]:
                model_bytes += param.numel() * 8

        print(f"Total model size: {model_bytes / 1e9:.4f} GB", file=stderr)

        # Move model to MPS device
        if device is None or device == 'auto':
            device = 'mps'
            print(f"Using Metal Performance Shaders (MPS) backend on device: {device}", file=stderr)
        
        if device != 'mps':
            print(f"Warning: Device '{device}' specified, but only 'mps' is supported. Using 'mps'.", file=stderr)
            device = 'mps'
        
        model = model.to(device)

        # Verify all parameters are on MPS
        cpu_params = sum(1 for p in model.parameters() if p.device.type == 'cpu')
        if cpu_params > 0:
            print(f"Warning: {cpu_params} parameters are still on CPU. "
                  f"Model may not fit in memory.", file=stderr)

        return model