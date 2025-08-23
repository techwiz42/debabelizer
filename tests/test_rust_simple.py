#!/usr/bin/env python3
"""Simple test of Rust implementation with generated audio data."""

import asyncio
import os
import sys
import numpy as np

# Add debabelizer-python to path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

try:
    import debabelizer
    print("✅ Successfully imported Rust-based debabelizer")
except ImportError as e:
    print(f"❌ Failed to import debabelizer: {e}")
    sys.exit(1)

def generate_speech_like_audio(duration=3.0, sample_rate=16000):
    """Generate speech-like audio data."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like formant frequencies
    f1, f2, f3 = 700, 1220, 2600  # Typical vowel formants
    
    # Generate base signal with harmonics
    audio = np.zeros_like(t)
    
    # Add fundamental and harmonics
    fundamental = 150  # Hz
    for harmonic in range(1, 6):
        freq = fundamental * harmonic
        if freq < sample_rate / 2:  # Nyquist limit
            audio += (1.0 / harmonic) * np.sin(2 * np.pi * freq * t)
    
    # Add formant resonances
    for formant in [f1, f2, f3]:
        audio += 0.3 * np.sin(2 * np.pi * formant * t) * np.exp(-t * 2)
    
    # Add envelope to make it more natural
    envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
    audio *= envelope
    
    # Normalize and convert to 16-bit PCM
    audio = audio / np.max(np.abs(audio)) * 0.7  # Prevent clipping
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

async def test_rust_implementation():
    """Test the Rust implementation with generated audio."""
    print("🚀 Testing Rust Debabelizer Implementation\n")
    
    # Check API key
    if not os.getenv("SONIOX_API_KEY"):
        print("⚠️  SONIOX_API_KEY not set. Using dummy key for connection test...")
        os.environ["SONIOX_API_KEY"] = "dummy-key-for-testing"
    
    # Generate test audio
    print("🎵 Generating speech-like test audio...")
    audio_data = generate_speech_like_audio(duration=2.0)
    print(f"✅ Generated {len(audio_data)} bytes of audio data")
    
    # Create audio format and data objects
    try:
        audio_format = debabelizer.AudioFormat("pcm", 16000, 1, 16)
        audio = debabelizer.AudioData(audio_data, audio_format)
        print(f"✅ Created audio objects (format: {audio_format.format}, rate: {audio_format.sample_rate})")
    except Exception as e:
        print(f"❌ Failed to create audio objects: {e}")
        return
    
    # Create processor with Soniox configuration
    try:
        config = {
            "preferences": {
                "stt_provider": "soniox",
                "auto_select": False
            },
            "soniox": {
                "api_key": os.getenv("SONIOX_API_KEY"),
                "model": "stt-rt-preview",
                "auto_detect_language": True
            }
        }
        
        processor = debabelizer.VoiceProcessor(config=config)
        print(f"✅ Created processor with provider: {processor.current_stt_provider}")
    except Exception as e:
        print(f"❌ Failed to create processor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test batch transcription
    print("\n📝 Testing Batch Transcription:")
    try:
        result = processor.transcribe(audio)
        print(f"✅ Transcription completed:")
        print(f"   Text: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2f}")
        if result.language_detected:
            print(f"   Language: {result.language_detected}")
    except Exception as e:
        print(f"❌ Batch transcription failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test streaming transcription
    print("\n🎙️  Testing Streaming Transcription:")
    try:
        # Start streaming session
        session_id = await processor.start_streaming_transcription()
        print(f"✅ Started streaming session: {session_id}")
        
        # Create result collection task
        results = []
        
        async def collect_results():
            try:
                timeout_count = 0
                async for result in processor.get_streaming_results(session_id, timeout=3.0):
                    print(f"   📝 Result {len(results)+1}: text='{result.text}', is_final={result.is_final}, confidence={result.confidence:.2f}")
                    results.append(result)
                    
                    # Reset timeout counter on actual results
                    if result.text.strip():
                        timeout_count = 0
                    else:
                        timeout_count += 1
                        
                    # Exit if too many empty results (likely connection issue)
                    if timeout_count > 5:
                        print("   ⚠️  Too many empty results, likely connection issue")
                        break
                        
            except Exception as e:
                print(f"   ❌ Result collection error: {e}")
        
        # Start collecting results
        results_task = asyncio.create_task(collect_results())
        
        # Send audio in chunks
        chunk_size = 3200  # 0.2 seconds of 16kHz mono
        total_chunks = len(audio_data) // chunk_size
        print(f"📤 Sending {total_chunks} audio chunks ({chunk_size} bytes each)...")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if chunk:
                await processor.stream_audio(session_id, chunk)
                print(f"   📤 Sent chunk {i//chunk_size + 1}/{total_chunks}")
                await asyncio.sleep(0.2)  # Simulate real-time streaming
        
        print("✅ All audio chunks sent, waiting for final results...")
        
        # Wait a bit for final processing
        await asyncio.sleep(2.0)
        
        # Stop streaming
        await processor.stop_streaming_transcription(session_id)
        print("✅ Streaming session stopped")
        
        # Wait for results collection to finish
        await results_task
        
        # Summary
        final_text = " ".join(r.text for r in results if r.is_final and r.text.strip())
        if final_text:
            print(f"\n📝 Final transcription: '{final_text}'")
        else:
            print(f"\n⚠️  No final transcription received (got {len(results)} results total)")
        print(f"   Total results received: {len(results)}")
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function."""
    print("=" * 60)
    print("RUST DEBABELIZER VOICE DATA TEST")
    print("=" * 60)
    
    await test_rust_implementation()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())