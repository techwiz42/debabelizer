#!/usr/bin/env python3
"""
Test Soniox streaming with realistic audio data that matches frontend format
"""

import asyncio
import sys
import numpy as np
import struct

try:
    import debabelizer
    print("✓ Successfully imported debabelizer")
except ImportError as e:
    print(f"✗ Failed to import debabelizer: {e}")
    sys.exit(1)

def generate_test_audio():
    """Generate test audio similar to what the frontend sends"""
    # Generate a simple tone at 440Hz (A note) for 2 seconds
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440.0  # Hz
    
    # Generate samples
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave with some variation to make it more speech-like
    audio_signal = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some noise and amplitude variation to make it more realistic
    noise = np.random.normal(0, 0.05, len(audio_signal))
    audio_signal = audio_signal + noise
    
    # Apply some amplitude modulation to simulate speech-like patterns
    envelope = np.sin(2 * np.pi * 2 * t) * 0.5 + 0.5  # 2Hz modulation
    audio_signal = audio_signal * envelope
    
    # Convert to 16-bit PCM (signed integers)
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    
    # Convert to bytes (little-endian)
    audio_bytes = audio_int16.tobytes()
    
    print(f"Generated {len(audio_bytes)} bytes of test audio ({duration}s at {sample_rate}Hz)")
    return audio_bytes

def chunk_audio(audio_bytes, chunk_size=170):
    """Split audio into chunks similar to what the frontend sends"""
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def test_soniox_with_real_audio():
    """Test Soniox streaming with realistic audio data"""
    
    print("Creating VoiceProcessor with Soniox...")
    
    # Create config for Soniox
    config_dict = {
        "soniox": {
            "api_key": "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"
        },
        "preferences": {
            "stt_provider": "soniox"
        }
    }
    
    try:
        config = debabelizer.DebabelizerConfig(config_dict)
        processor = debabelizer.VoiceProcessor(config=config)
        print("✓ VoiceProcessor created successfully")
        
        # Generate test audio
        print("Generating test audio...")
        audio_data = generate_test_audio()
        audio_chunks = chunk_audio(audio_data, chunk_size=170)
        print(f"Split into {len(audio_chunks)} chunks of ~170 bytes each")
        
        # Start streaming session
        print("Starting Soniox streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            interim_results=True
        )
        print(f"✓ Streaming session started: {session_id}")
        
        # Get streaming results iterator
        print("Getting streaming results iterator...")
        results_iter = processor.get_streaming_results(session_id)
        print("✓ Results iterator created successfully")
        
        # Create task to collect results
        results_collected = []
        
        async def collect_results():
            try:
                result_count = 0
                async for result in results_iter:
                    result_count += 1
                    results_collected.append(result)
                    print(f"📝 Result {result_count}: text='{result.text}', is_final={result.is_final}, confidence={result.confidence:.2f}")
                    
                    # Stop after getting 10 results or if we get a final result
                    if result_count >= 10 or result.is_final:
                        break
                        
                print(f"Results collection ended after {result_count} results")
            except Exception as e:
                print(f"Results collection error: {type(e).__name__}: {e}")
        
        # Start results collection task
        results_task = asyncio.create_task(collect_results())
        
        # Send audio chunks with realistic timing
        print(f"Sending {len(audio_chunks)} audio chunks...")
        for i, chunk in enumerate(audio_chunks):
            try:
                await processor.stream_audio(session_id, chunk)
                if i % 10 == 0:  # Log every 10th chunk
                    print(f"  Sent chunk {i+1}/{len(audio_chunks)} ({len(chunk)} bytes)")
                
                # Wait a bit between chunks to simulate real-time audio (170 bytes = ~10ms at 16kHz)
                await asyncio.sleep(0.01)  # 10ms delay
                
            except Exception as e:
                print(f"Error sending chunk {i}: {e}")
                break
        
        print("✓ All audio chunks sent")
        
        # Wait for results for a bit
        print("Waiting for transcription results (5 seconds)...")
        try:
            await asyncio.wait_for(results_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("Timeout waiting for results")
            results_task.cancel()
        
        # Stop streaming
        print("Stopping streaming session...")
        await processor.stop_streaming_transcription(session_id)
        print("✓ Streaming session stopped")
        
        # Report results
        print(f"\n📊 Summary:")
        print(f"   Audio sent: {len(audio_data)} bytes ({len(audio_chunks)} chunks)")
        print(f"   Results received: {len(results_collected)}")
        
        if results_collected:
            print("   Transcription results:")
            for i, result in enumerate(results_collected):
                print(f"     {i+1}. '{result.text}' (final={result.is_final}, conf={result.confidence:.2f})")
            return True
        else:
            print("   ❌ No transcription results received")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Soniox Real Audio Streaming Test")
    print("=" * 45)
    
    success = asyncio.run(test_soniox_with_real_audio())
    
    if success:
        print("\n✅ Test completed - Soniox responded to audio!")
        sys.exit(0)
    else:
        print("\n❌ Test failed - No transcription results")
        sys.exit(1)