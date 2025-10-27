#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def quick_test():
    """Quick test to verify the fix works"""
    print("🔧 QUICK TEST OF THE FIX")
    print("=" * 40)
    
    try:
        from vsd_net import ASD_Wrapper_Visual_Encoder
        
        # Test the exact case that was failing
        B, N, T = 1, 2, 10
        visual_model = ASD_Wrapper_Visual_Encoder(num_speakers=N)
        video_input = torch.randn(B * N, T, 96, 96)
        
        print(f"Input: {video_input.shape}")
        _, output = visual_model(video_input, return_embedding=True)
        print(f"Output: {output.shape}")
        
        expected = (B, N, T, 256)
        if output.shape == expected:
            print("✅ FIX WORKING! Division by zero resolved.")
            return True
        else:
            print(f"❌ Shape mismatch: got {output.shape}, expected {expected}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if quick_test():
        print("\n🎉 The visual encoder fix is working!")
        print("You can now run your training pipeline.")
    else:
        print("\n💥 The fix needs more work.")