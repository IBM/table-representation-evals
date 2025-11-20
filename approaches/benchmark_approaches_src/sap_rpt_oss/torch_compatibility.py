#!/usr/bin/env python3
"""
PyTorch Compatibility Layer for SAP RPT-1-OSS

This module provides compatibility for PyTorch versions that don't have torch.nn.attention
by creating a mock implementation of the required modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from contextlib import nullcontext

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
        """Mock sdpa_kernel function for compatibility - returns a null context manager"""
        return nullcontext()
    
    # Monkey patch torch.nn to add attention as a proper module
    # This must be done before sap_rpt_oss imports torch.nn.attention
    import torch.nn as nn_module
    import types
    import sys
    
    if not hasattr(nn_module, 'attention'):
        # Create a proper module object that can be imported
        attention_module = types.ModuleType('attention')
        attention_module.SDPBackend = SDPBackend
        attention_module.sdpa_kernel = sdpa_kernel
        attention_module.__all__ = ['SDPBackend', 'sdpa_kernel']
        
        # Add to torch.nn
        nn_module.attention = attention_module
        
        # Also register it in sys.modules so 'from torch.nn.attention import ...' works
        sys.modules['torch.nn.attention'] = attention_module

print(f"PyTorch compatibility layer loaded. torch.nn.attention available: {hasattr(torch.nn, 'attention')}")

