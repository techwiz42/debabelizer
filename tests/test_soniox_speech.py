#!/usr/bin/env python3
"""
Test Soniox streaming with speech-like audio that should trigger transcription
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

def generate_speech_like_audio():
    """Generate speech-like audio that should trigger Soniox transcription"""
    # Parameters for speech-like audio
    sample_rate = 16000
    duration = 3.0  # 3 seconds of "speech"
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a more complex waveform that resembles speech patterns
    # Combine multiple frequencies that are common in human speech (formants)
    
    # Base frequencies for vowel-like sounds (Hz)
    f1 = 350  # First formant (roughly "ah" sound)
    f2 = 1200  # Second formant
    f3 = 2400  # Third formant
    
    # Create the base signal with formant-like structure
    signal = (
        np.sin(2 * np.pi * f1 * t) * 0.4 +
        np.sin(2 * np.pi * f2 * t) * 0.3 +
        np.sin(2 * np.pi * f3 * t) * 0.2
    )
    
    # Add amplitude modulation to simulate speech rhythm (syllables)
    # Typical speech has about 3-5 syllables per second
    syllable_rate = 4.0  # syllables per second
    envelope = np.abs(np.sin(2 * np.pi * syllable_rate * t))
    
    # Apply envelope to create speech-like bursts
    signal = signal * envelope
    
    # Add some higher frequency content (consonant-like sounds)
    consonant_freq = 4000
    consonant_signal = np.sin(2 * np.pi * consonant_freq * t) * 0.1
    # Make consonants appear briefly between vowels
    consonant_envelope = np.abs(np.sin(2 * np.pi * syllable_rate * 2 * t + np.pi/2))
    consonant_envelope = np.where(consonant_envelope > 0.8, consonant_envelope, 0)
    signal += consonant_signal * consonant_envelope
    
    # Add realistic noise and dynamic range
    noise = np.random.normal(0, 0.02, len(signal))
    signal = signal + noise
    
    # Apply realistic amplitude variation (speech is not constant volume)
    volume_variation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Slow volume changes
    signal = signal * volume_variation
    
    # Normalize and convert to 16-bit PCM
    signal = np.clip(signal, -1.0, 1.0)  # Clip to prevent overflow
    audio_int16 = (signal * 20000).astype(np.int16)  # Use 20000 instead of 32767 for realistic speech levels
    
    # Convert to bytes (little-endian)
    audio_bytes = audio_int16.tobytes()
    
    print(f"Generated {len(audio_bytes)} bytes of speech-like audio ({duration}s at {sample_rate}Hz)")
    print(f"Audio contains formant frequencies: {f1}Hz, {f2}Hz, {f3}Hz with {syllable_rate} syllables/sec")
    return audio_bytes

def chunk_audio(audio_bytes, chunk_size=170):
    """Split audio into chunks similar to what the frontend sends"""
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def test_soniox_with_speech():
    """Test Soniox streaming with speech-like audio"""
    
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
        
        # Generate speech-like audio
        print("Generating speech-like audio...")
        audio_data = generate_speech_like_audio()
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
                    print(f"🎤 Result {result_count}: text='{result.text}', is_final={result.is_final}, confidence={result.confidence:.2f}")
                    
                    # Stop after getting 20 results or if we get 2 final results
                    if result_count >= 20 or (result.is_final and len([r for r in results_collected if r.is_final]) >= 2):
                        break
                        
                print(f"Results collection ended after {result_count} results")
            except Exception as e:
                print(f"Results collection error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        # Start results collection task
        results_task = asyncio.create_task(collect_results())
        
        # Send audio chunks with realistic timing
        print(f"Sending {len(audio_chunks)} audio chunks...")
        chunks_sent = 0
        for i, chunk in enumerate(audio_chunks):
            try:
                await processor.stream_audio(session_id, chunk)
                chunks_sent += 1
                
                if i % 20 == 0:  # Log every 20th chunk
                    print(f"  Sent chunk {i+1}/{len(audio_chunks)} ({len(chunk)} bytes)")
                
                # Wait a bit between chunks to simulate real-time audio (170 bytes = ~10ms at 16kHz)
                await asyncio.sleep(0.01)  # 10ms delay
                
            except Exception as e:
                print(f"Error sending chunk {i}: {e}")
                break
        
        print(f"✓ Sent {chunks_sent}/{len(audio_chunks)} audio chunks successfully")
        
        # Wait longer for transcription results since we have real speech
        print("Waiting for transcription results (10 seconds)...")
        try:
            await asyncio.wait_for(results_task, timeout=10.0)
        except asyncio.TimeoutError:
            print("Timeout waiting for results")
            results_task.cancel()
        
        # Stop streaming
        print("Stopping streaming session...")
        await processor.stop_streaming_transcription(session_id)
        print("✓ Streaming session stopped")
        
        # Report results
        print(f"\n📊 Summary:")
        print(f"   Audio sent: {len(audio_data)} bytes ({chunks_sent} chunks)")
        print(f"   Results received: {len(results_collected)}")
        
        if results_collected:
            print("   Transcription results:")
            for i, result in enumerate(results_collected):
                status = "FINAL" if result.is_final else "interim"
                print(f"     {i+1}. [{status}] '{result.text}' (conf={result.confidence:.2f})")
            
            # Check if we got meaningful transcription
            final_results = [r for r in results_collected if r.is_final and r.text.strip()]
            if final_results:
                print(f"\n🎉 SUCCESS: Got {len(final_results)} final transcription(s)!")
                return True
            else:
                print(f"\n⚠️  Got results but no final transcriptions with text")
                return len(results_collected) > 0
        else:
            print("   ❌ No transcription results received")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Soniox Speech-Like Audio Test")
    print("=" * 40)
    print("Generating audio with speech formants and syllable patterns...")
    
    success = asyncio.run(test_soniox_with_speech())
    
    if success:
        print("\n✅ Test completed - Soniox responded to speech-like audio!")
        sys.exit(0)
    else:
        print("\n❌ Test failed - No meaningful transcription results")
        sys.exit(1)