#!/usr/bin/env python3
"""
PyTorch Compatibility Layer for ConTextTab

This module provides compatibility for PyTorch versions that don't have torch.nn.attention
by creating a mock implementation of the required modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

# Check if torch.nn.attention exists
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    HAS_ATTENTION = True
except ImportError:
    HAS_ATTENTION = False

if not HAS_ATTENTION:
    # Create mock implementations for PyTorch < 2.7.0
    class SDPBackend:
        """Mock SDPBackend for compatibility"""
        FLASH_ATTENTION = "flash_attention"
        MATH = "math"
        EFFICIENT_ATTENTION = "efficient_attention"
        
        @staticmethod
        def get_backend(attn_bias, query, key, value, is_causal=False, dropout_p=0.0, scale=None):
            return SDPBackend.MATH
    
    def sdpa_kernel(*args, **kwargs):
        """Mock sdpa_kernel function for compatibility"""
        return None
    
    # Monkey patch torch.nn.attention
    import torch.nn
    torch.nn.attention = type('attention', (), {
        'SDPBackend': SDPBackend,
        'sdpa_kernel': sdpa_kernel
    })()

print(f"PyTorch compatibility layer loaded. torch.nn.attention available: {HAS_ATTENTION}")
