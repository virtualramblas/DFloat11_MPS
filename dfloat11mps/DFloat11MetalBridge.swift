// DFloat11MetalBridge.swift
// Swift wrapper for Metal kernel execution with proper buffer management
// This can be compiled as a C-compatible library for Python

import Metal
import Darwin

// MARK: - C-compatible interface

/// Initialize Metal device and compile kernel
/// Returns an opaque pointer to the decoder instance
@_cdecl("dfloat11_create_decoder")
public func dfloat11_create_decoder() -> UnsafeMutableRawPointer? {
    guard let decoder = DFloat11MetalDecoder() else {
        return nil
    }
    let pointer = Unmanaged.passRetained(decoder).toOpaque()
    return pointer
}

/// Release the decoder instance
@_cdecl("dfloat11_release_decoder")
public func dfloat11_release_decoder(_ decoder: UnsafeMutableRawPointer?) {
    guard let decoder = decoder else { return }
    Unmanaged<DFloat11MetalDecoder>.fromOpaque(decoder).release()
}

/// Execute the decode kernel
/// All buffer pointers are raw pointers to data from PyTorch/NumPy
@_cdecl("dfloat11_decode")
public func dfloat11_decode(
    _ decoder: UnsafeMutableRawPointer?,
    _ luts: UnsafeRawPointer?,
    _ luts_size: Int,
    _ encoded: UnsafeRawPointer?,
    _ encoded_size: Int,
    _ sign_mantissa: UnsafeRawPointer?,
    _ sign_size: Int,
    _ output_positions: UnsafeRawPointer?,
    _ outpos_size: Int,
    _ gaps: UnsafeRawPointer?,
    _ gaps_size: Int,
    _ output: UnsafeMutableRawPointer?,
    _ output_size: Int,
    _ n_luts: UInt32,
    _ n_bytes: UInt32,
    _ n_elements: UInt32,
    _ shared_mem_size: Int,
    _ threads_per_block: Int
) -> Int32 {
    guard let decoder = decoder,
          let luts = luts,
          let encoded = encoded,
          let sign_mantissa = sign_mantissa,
          let output_positions = output_positions,
          let gaps = gaps,
          let output = output else {
        return -1
    }
    
    let decoderObj = Unmanaged<DFloat11MetalDecoder>.fromOpaque(decoder).takeUnretainedValue()
    
    do {
        try decoderObj.decode(
            luts: luts, lutsSize: luts_size,
            encoded: encoded, encodedSize: encoded_size,
            signMantissa: sign_mantissa, signSize: sign_size,
            outputPositions: output_positions, outposSize: outpos_size,
            gaps: gaps, gapsSize: gaps_size,
            output: output, outputSize: output_size,
            nLuts: n_luts,
            nBytes: n_bytes,
            nElements: n_elements,
            sharedMemSize: shared_mem_size,
            threadsPerBlock: threads_per_block
        )
        return 0
    } catch {
        print("Decode error: \(error)")
        return -2
    }
}

// MARK: - Metal Decoder Implementation

class DFloat11MetalDecoder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState
    
    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return nil
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = commandQueue
        
        // Compile the Metal kernel
        let kernelSource = """
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
        
        do {
            let library = try device.makeLibrary(source: kernelSource, options: nil)
            guard let function = library.makeFunction(name: "dfloat11_decode") else {
                print("Failed to find kernel function")
                return nil
            }
            
            self.pipeline = try device.makeComputePipelineState(function: function)
            print("Metal decoder initialized successfully")
        } catch {
            print("Failed to compile Metal kernel: \(error)")
            return nil
        }
    }
    
    func decode(
        luts: UnsafeRawPointer, lutsSize: Int,
        encoded: UnsafeRawPointer, encodedSize: Int,
        signMantissa: UnsafeRawPointer, signSize: Int,
        outputPositions: UnsafeRawPointer, outposSize: Int,
        gaps: UnsafeRawPointer, gapsSize: Int,
        output: UnsafeMutableRawPointer, outputSize: Int,
        nLuts: UInt32,
        nBytes: UInt32,
        nElements: UInt32,
        sharedMemSize: Int,
        threadsPerBlock: Int
    ) throws {
        print("Swift decode called with:")
        print("  n_luts: \(nLuts), n_bytes: \(nBytes), n_elements: \(nElements)")
        print("  Buffer sizes - luts: \(lutsSize), encoded: \(encodedSize), sign: \(signSize)")
        print("  threads_per_block: \(threadsPerBlock), shared_mem: \(sharedMemSize)")
        
        // Create Metal buffers from raw pointers
        guard let lutsBuffer = device.makeBuffer(bytes: luts, length: lutsSize, options: .storageModeShared) else {
            print("Failed to create luts buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        guard let encodedBuffer = device.makeBuffer(bytes: encoded, length: encodedSize, options: .storageModeShared) else {
            print("Failed to create encoded buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        guard let signBuffer = device.makeBuffer(bytes: signMantissa, length: signSize, options: .storageModeShared) else {
            print("Failed to create sign buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        // Output positions buffer must be mutable (kernel writes to it)
        // Copy the input data and create a writable buffer
        guard let outposBuffer = device.makeBuffer(bytes: outputPositions, length: outposSize, options: .storageModeShared) else {
            print("Failed to create output positions buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        guard let gapsBuffer = device.makeBuffer(bytes: gaps, length: gapsSize, options: .storageModeShared) else {
            print("Failed to create gaps buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            print("Failed to create output buffer")
            throw DecoderError.bufferCreationFailed
        }
        
        print("All Metal buffers created successfully")
        
        print("All Metal buffers created successfully")
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Failed to create command buffer")
            throw DecoderError.commandBufferCreationFailed
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create compute encoder")
            throw DecoderError.commandBufferCreationFailed
        }
        
        print("Command buffer and encoder created")
        
        encoder.setComputePipelineState(pipeline)
        
        // Set buffers
        encoder.setBuffer(lutsBuffer, offset: 0, index: 0)
        encoder.setBuffer(encodedBuffer, offset: 0, index: 1)
        encoder.setBuffer(signBuffer, offset: 0, index: 2)
        encoder.setBuffer(outposBuffer, offset: 0, index: 3)
        encoder.setBuffer(gapsBuffer, offset: 0, index: 4)
        encoder.setBuffer(outputBuffer, offset: 0, index: 5)
        
        print("Buffers set")
        
        // Set constants
        var nLutsCopy = nLuts
        var nBytesCopy = nBytes
        var nElementsCopy = nElements
        encoder.setBytes(&nLutsCopy, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&nBytesCopy, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&nElementsCopy, length: MemoryLayout<UInt32>.size, index: 8)
        
        print("Constants set")
        
        // Set threadgroup memory
        encoder.setThreadgroupMemoryLength(sharedMemSize, index: 0)
        
        print("Threadgroup memory set: \(sharedMemSize) bytes")
        
        // Calculate grid size
        let blocksPerGrid = (Int(nBytes) + (threadsPerBlock * 8) - 1) / (threadsPerBlock * 8)
        let totalThreads = blocksPerGrid * threadsPerBlock
        
        print("Grid configuration: \(blocksPerGrid) blocks, \(threadsPerBlock) threads/block, \(totalThreads) total")
        
        let gridSize = MTLSize(width: totalThreads, height: 1, depth: 1)
        let threadgroupSize = MTLSize(width: threadsPerBlock, height: 1, depth: 1)
        
        // Dispatch
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        print("Kernel dispatched, waiting for completion...")
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        print("Kernel execution completed with status: \(commandBuffer.status.rawValue)")
        
        // Check for errors
        if let error = commandBuffer.error {
            print("Kernel execution error: \(error.localizedDescription)")
            throw DecoderError.executionFailed(error.localizedDescription)
        }
        
        if commandBuffer.status == .error {
            print("Command buffer status is error")
            throw DecoderError.executionFailed("Command buffer execution failed")
        }
        
        print("Copying output buffer (\(outputSize) bytes) back to host memory...")
        
        // Copy output buffer back to the provided pointer
        let outputPointer = outputBuffer.contents()
        memcpy(output, outputPointer, outputSize)
        
        print("Decode completed successfully")
    }
}

enum DecoderError: Error {
    case bufferCreationFailed
    case commandBufferCreationFailed
    case executionFailed(String)
}