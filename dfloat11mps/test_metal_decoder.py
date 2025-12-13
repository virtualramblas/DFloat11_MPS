import numpy as np
import torch
from dfloat11_metal_wrapper import MetalDecoder

def test_decoder():
    decoder = MetalDecoder()
    
    # Create dummy test data
    luts = np.random.randint(0, 256, size=(5, 256), dtype=np.uint8)
    encoded = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    sign = np.random.randint(0, 256, size=1000, dtype=np.uint8)
    outpos = np.array([0, 500, 1000], dtype=np.uint32)
    gaps = np.random.randint(0, 32, size=625, dtype=np.uint8)
    output = np.zeros(1000, dtype=np.uint16)
    
    # Run decoder
    try:
        decoder.decode(
            luts, encoded, sign, outpos, gaps, output,
            n_luts=5, n_bytes=1000, n_elements=1000,
            shared_mem_size=8192, threads_per_block=256
        )
        print("✓ Decoder executed successfully")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min()}, {output.max()}]")
    except Exception as e:
        print(f"✗ Decoder failed: {e}")
        raise

if __name__ == "__main__":
    test_decoder()