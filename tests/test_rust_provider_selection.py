#!/usr/bin/env python3

"""
Test script to verify if Rust Soniox provider is being selected and used.
This will help identify if the Rust implementation is being called or if 
legacy Python code is being invoked instead.
"""

import os
import sys
import asyncio

# Set up environment for Soniox
os.environ["SONIOX_API_KEY"] = "fake_key_for_testing"
os.environ["DEBABELIZER_STT_PROVIDER"] = "soniox"

try:
    print("🔍 Importing debabelizer...")
    import debabelizer
    print("✅ Successfully imported debabelizer")
    
    print("\n🔍 Creating VoiceProcessor...")
    processor = debabelizer.VoiceProcessor(stt_provider="soniox")
    print("✅ Successfully created VoiceProcessor")
    
    # Try to trigger provider initialization
    print("\n🔍 Attempting to get available providers...")
    try:
        # This should trigger provider initialization in Rust
        print("🔍 Calling processor methods to trigger initialization...")
        
        # Try a simple transcription that should trigger the provider
        dummy_audio = b'\x00' * 1000  # 1000 bytes of silence
        audio_format = debabelizer.AudioFormat("wav", 16000, 1, 16)
        audio_data = debabelizer.AudioData(dummy_audio, audio_format)
        
        print("🔍 Attempting transcription (this should trigger Rust provider)...")
        try:
            result = processor.transcribe(audio_data)
            print(f"✅ Transcription completed: {result}")
        except Exception as e:
            print(f"⚠️  Transcription failed (expected with fake API key): {e}")
            print("   But this should have shown Rust debug output if it was called...")
        
    except Exception as e:
        print(f"❌ Error during provider operations: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("If you see '🔍 RUST: ensure_initialized() called' above, Rust is working.")
    print("If you DON'T see that message, the legacy Python implementation is being used.")
    print("="*60)
    
except ImportError as e:
    print(f"❌ Failed to import debabelizer: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)