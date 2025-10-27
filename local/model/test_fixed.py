#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_visual_encoder_fixed():
    """Test the fixed visual encoder"""
    print("🧪 TESTING FIXED VISUAL ENCODER")
    print("=" * 50)
    
    try:
        from vsd_net import ASD_Wrapper_Visual_Encoder
        
        # Test with different batch sizes
        test_cases = [
            (2, 4, 10),  # B=2, N=4, T=10
            (1, 4, 5),   # B=1, N=4, T=5
            (3, 4, 20),  # B=3, N=4, T=20
        ]
        
        for B, N, T in test_cases:
            print(f"\nTesting B={B}, N={N}, T={T}")
            
            # Create model with specified number of speakers
            visual_model = ASD_Wrapper_Visual_Encoder(num_speakers=N)
            
            # Create input
            video_input = torch.randn(B * N, T, 96, 96)
            print(f"Input shape: {video_input.shape}")
            
            # Forward pass
            _, output = visual_model(video_input, return_embedding=True)
            print(f"Output shape: {output.shape}")
            
            # Verify output shape
            expected_shape = (B, N, T, 256)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            print("✅ Shape correct!")
        
        print("\n🎉 VISUAL ENCODER FIXED AND WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ VISUAL ENCODER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline_fixed():
    """Test the complete pipeline with fixed visual encoder"""
    print("\n🧪 TESTING COMPLETE PIPELINE WITH FIX")
    print("=" * 50)
    
    try:
        from vsd_net import ASD_Wrapper_Visual_Encoder
        from casa_net import CASA_Module
        from avsd_net import Audio_Feature_Extractor
        
        # Test configuration
        B, N, T = 2, 4, 10
        Freq = 40
        
        print("1. Testing Visual Encoder...")
        visual_model = ASD_Wrapper_Visual_Encoder(num_speakers=N)
        video_input = torch.randn(B * N, T, 96, 96)
        _, visual_output = visual_model(video_input, return_embedding=True)
        print(f"   Visual: {video_input.shape} -> {visual_output.shape}")
        assert visual_output.shape == (B, N, T, 256)
        print("   ✅ Visual encoder: CORRECT")
        
        print("\n2. Testing Audio Encoder...")
        audio_model = Audio_Feature_Extractor(input_dim=40, audio_output_dim=128)
        audio_input = torch.randn(B, 2, Freq, T)
        audio_output = audio_model(audio_input)
        print(f"   Audio: {audio_input.shape} -> {audio_output.shape}")
        assert audio_output.shape == (B, T, 128)
        print("   ✅ Audio encoder: CORRECT")
        
        print("\n3. Testing CASA Module...")
        casa_model = CASA_Module(dim_audio=128, dim_video=256, dim_spk=100, num_heads=2)
        
        # Prepare inputs for CASA (expand audio for all speakers)
        audio_features = audio_output.unsqueeze(-1).expand(-1, -1, -1, N)  # (B, T, 128, N)
        video_features = visual_output.permute(0, 2, 3, 1)  # (B, T, 256, N)
        speaker_embeddings = torch.randn(B, T, 100, N)
        
        casa_output = casa_model(audio_features, video_features, speaker_embeddings)
        print(f"   CASA Input - Audio: {audio_features.shape}, Video: {video_features.shape}")
        print(f"   CASA Output: {casa_output.shape}")
        
        expected_shape = (B, T, 128+256+100, N)
        assert casa_output.shape == expected_shape
        print("   ✅ CASA module: CORRECT")
        
        print("\n🎉 COMPLETE PIPELINE WORKING!")
        return True
        
    except Exception as e:
        print(f"❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 TESTING WITH VISUAL ENCODER FIX")
    print("This should resolve the division by zero error\n")
    
    # Test the fixed visual encoder
    visual_ok = test_visual_encoder_fixed()
    
    # Test complete pipeline
    pipeline_ok = False
    if visual_ok:
        pipeline_ok = test_complete_pipeline_fixed()
    
    print("\n" + "=" * 50)
    if visual_ok and pipeline_ok:
        print("🎉 ALL TESTS PASSED WITH THE FIX!")
        print("The division by zero error has been resolved.")
        print("Your CASA-Net implementation is now READY FOR TRAINING! 🚀")
    else:
        print("❌ TESTS FAILED")
        print("The fix needs more work")
    
    return visual_ok and pipeline_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)