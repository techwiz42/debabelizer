#!/usr/bin/env python3

import sys
import os
import asyncio
import traceback

# Add the built target directory to Python path
target_dir = "/home/peter/debabelizer/target/release"
if os.path.exists(target_dir):
    sys.path.insert(0, target_dir)

try:
    # Try to import the compiled library directly
    import _internal as debabelizer_rust
    print("✅ Successfully imported compiled Rust library")
except ImportError as e:
    print(f"❌ Failed to import compiled library: {e}")
    
    # Fall back to trying the installed package
    try:
        import debabelizer
        print("✅ Using installed debabelizer package")
    except ImportError as e2:
        print(f"❌ No debabelizer package available: {e2}")
        sys.exit(1)

async def test_provider_initialization():
    """Test to see if our debug messages appear during provider initialization"""
    
    print("🚀 Starting provider initialization test...")
    print("=" * 60)
    
    try:
        print("1. Creating VoiceProcessor with Soniox (this should trigger provider init)...")
        if 'debabelizer_rust' in locals():
            processor = debabelizer_rust.VoiceProcessor(stt_provider="soniox")
        else:
            # Use the current installed version API
            processor = debabelizer.VoiceProcessor(stt_provider="soniox")
        print("✅ VoiceProcessor created")
        
        print("\n2. Attempting to start streaming (this should trigger provider init)...")
        session_id = await processor.start_streaming_transcription(
            audio_format="wav", 
            sample_rate=16000
        )
        print(f"✅ Streaming session started: {session_id}")
        
        print("\n3. Cleaning up...")
        await processor.stop_streaming_transcription(session_id)
        print("✅ Session stopped")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_provider_initialization())