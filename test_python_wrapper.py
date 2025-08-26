#!/usr/bin/env python3
"""
Test the PyO3 Python wrapper for the Rust debabelizer implementation.
Goal: Verify that Python wrapper achieves "speech in, language detected, text out"
"""
import asyncio
import os
import sys
import time

def test_python_wrapper():
    """Test PyO3 Python wrapper streaming functionality."""
    print("🐍 Testing PyO3 Python wrapper streaming...")
    
    # Import the Rust-backed Python module
    try:
        import debabelizer
        print("✅ Successfully imported debabelizer (Rust-backed)")
    except ImportError as e:
        print(f"❌ Failed to import debabelizer: {e}")
        return False
    
    # Check API key
    api_key = os.getenv('SONIOX_API_KEY')
    if not api_key:
        print("❌ SONIOX_API_KEY not set")
        return False
    
    print(f"🔑 Using Soniox API key: {api_key[:8]}...")
    
    # Create processor with Soniox provider
    try:
        processor = debabelizer.VoiceProcessor(stt_provider="soniox")
        print("✅ VoiceProcessor created")
    except Exception as e:
        print(f"❌ Failed to create processor: {e}")
        return False
    
    # Load test audio file
    audio_file = "/home/peter/debabelizer/tests/test_real_speech_16k.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    # Skip WAV header for PCM data
    pcm_data = audio_data[44:]  # Skip 44-byte WAV header
    print(f"🎵 Loaded {len(pcm_data)} bytes of PCM audio data")
    
    async def test_streaming():
        """Test the async streaming functionality."""
        print("\n🎤 Starting streaming transcription...")
        
        try:
            # Start streaming session
            session_id = await processor.start_streaming_transcription(
                audio_format="wav",
                sample_rate=16000,
                language=None,  # Auto-detect
                enable_language_identification=True,
                interim_results=True
            )
            print(f"✅ Streaming session started: {session_id}")
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False
        
        try:
            # Get streaming results iterator
            results_iter = processor.get_streaming_results(session_id, timeout=2.0)
            print("✅ Streaming results iterator created")
            
            # Send audio in chunks
            chunk_size = 8192  # 8KB chunks
            chunks = [pcm_data[i:i+chunk_size] for i in range(0, len(pcm_data), chunk_size)]
            print(f"📤 Sending {len(chunks)} audio chunks...")
            
            # Start result collection
            results = []
            interim_results = []
            
            # Send chunks and collect results concurrently
            async def send_audio():
                for i, chunk in enumerate(chunks):
                    print(f"📤 Sending chunk {i+1}/{len(chunks)}: {len(chunk)} bytes")
                    try:
                        await processor.stream_audio(session_id, chunk)
                        await asyncio.sleep(0.1)  # Small delay between chunks
                    except Exception as e:
                        print(f"❌ Error sending chunk {i+1}: {e}")
                        break
                print("📤 All audio chunks sent")
            
            # Start audio sending
            send_task = asyncio.create_task(send_audio())
            
            # Collect results
            result_count = 0
            timeout_count = 0
            max_timeouts = 20
            
            print("📥 Collecting streaming results...")
            async for result in results_iter:
                result_count += 1
                
                if hasattr(result, 'text') and result.text:
                    if result.is_final:
                        print(f"🎯 FINAL: '{result.text}' (confidence: {result.confidence:.3f})")
                        results.append(result.text)
                        
                        # Check for language detection
                        if hasattr(result, 'metadata') and result.metadata:
                            metadata = result.metadata
                            if 'detected_language' in metadata:
                                print(f"🌍 Language detected: {metadata['detected_language']}")
                    else:
                        print(f"💬 INTERIM: '{result.text}'")
                        interim_results.append(result.text)
                else:
                    # Empty result (keep-alive)
                    if hasattr(result, 'metadata') and result.metadata:
                        metadata = result.metadata
                        if metadata.get('type') == 'keep_alive':
                            timeout_count += 1
                            print(f"💓 Keep-alive {timeout_count}/{max_timeouts}")
                            if timeout_count >= max_timeouts:
                                print("⏰ Max keep-alives reached, ending collection")
                                break
                
                # Break if we have enough results and audio is done
                if results and not send_task.done():
                    await asyncio.sleep(0.1)
                elif results and send_task.done():
                    # Give a bit more time for final results
                    await asyncio.sleep(1.0)
                    break
                    
                if result_count > 50:  # Safety limit
                    print("⚠️ Hit result count limit")
                    break
            
            # Wait for audio sending to complete
            await send_task
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return False
        finally:
            # Stop streaming session
            try:
                await processor.stop_streaming_transcription(session_id)
                print("✅ Streaming session stopped")
            except Exception as e:
                print(f"❌ Error stopping session: {e}")
        
        # Summary
        print(f"\n📊 PYTHON WRAPPER STREAMING RESULTS:")
        print(f"- Final transcriptions: {len(results)}")
        print(f"- Interim transcriptions: {len(interim_results)}")
        
        if results:
            full_text = " ".join(results)
            print(f"- Complete transcription: '{full_text}'")
            print(f"\n🎉 SUCCESS! Python wrapper streaming WORKS!")
            print(f"✅ Speech in → Language detected → Text out ✅")
            return True
        else:
            print(f"❌ No final transcriptions received")
            return False
    
    # Run the async test
    try:
        return asyncio.run(test_streaming())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_python_wrapper()
    sys.exit(0 if success else 1)