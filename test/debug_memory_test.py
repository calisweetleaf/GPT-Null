#!/usr/bin/env python3
"""
Minimal test to debug the MemoryTier comparison issue
"""

import torch
import traceback
from cas import neural_memory_runtime as nmr

try:
    # Create a simple config
    config = {
        'max_context_length': 1000,
        'memory_budget_mb': 100,
        'compression_target': 8.0,
        'max_tensors_per_tier': 1000,
        'compression_threshold': 0.5
    }
    
    print("Creating neural memory runtime...")
    runtime = nmr.NeuralMemoryRuntime(config)
    
    print("Creating test tensor...")
    test_tensor = torch.randn(100, 512)
    
    print("Calling store_activation...")
    tensor_id = runtime.store_activation("test_tensor", test_tensor, nmr.MemoryTier.HOT)
    
    print(f"Store successful, tensor_id: {tensor_id}")
    
    print("Calling retrieve_activation...")
    retrieved = runtime.retrieve_activation(tensor_id)
    
    print(f"Retrieve successful, retrieved tensor shape: {retrieved.shape if retrieved is not None else 'None'}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()