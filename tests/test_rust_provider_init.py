#!/usr/bin/env python3

"""
Test script to verify if Rust providers are being initialized during VoiceProcessor creation.
"""

import os
import sys

# Set up environment for Soniox
os.environ["SONIOX_API_KEY"] = "fake_key_for_testing"
os.environ["DEBABELIZER_STT_PROVIDER"] = "soniox"

try:
    print("🔍 Importing debabelizer...")
    import debabelizer
    print("✅ Successfully imported debabelizer")
    
    print("\n🔍 Creating VoiceProcessor with explicit soniox provider...")
    print("     This should trigger Rust provider initialization...")
    
    processor = debabelizer.VoiceProcessor(stt_provider="soniox")
    print("✅ Successfully created VoiceProcessor")
    
    # Try to use file transcription method which should work
    print("\n🔍 Testing file transcription (should trigger provider init)...")
    try:
        # Create a dummy audio file first
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write minimal WAV header + silence
            tmp_file.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            tmp_file_path = tmp_file.name
        
        print(f"🔍 Calling transcribe_file on {tmp_file_path}...")
        result = processor.transcribe_file(tmp_file_path)
        print(f"✅ Transcription result: {result}")
        
        # Clean up
        os.unlink(tmp_file_path)
        
    except Exception as e:
        print(f"⚠️  Transcription failed (expected with fake API key): {e}")
        print("   But we should have seen Rust debug output if provider init was called...")
    
    print("\n🔍 Getting provider name...")
    try:
        provider_name = processor.stt_provider_name
        print(f"✅ STT provider name: {provider_name}")
    except Exception as e:
        print(f"❌ Failed to get provider name: {e}")
    
    print("\n" + "="*70)
    print("EXPECTED RUST DEBUG OUTPUT:")
    print("🔍 RUST: ensure_initialized() called")
    print("🚀 RUST: Initializing providers for the first time...")
    print("🔍 RUST: Checking for Soniox provider config...")
    print("✅ RUST: Found Soniox provider config...")
    print("")
    print("If you DON'T see these messages above, the Rust code is not being executed!")
    print("="*70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()