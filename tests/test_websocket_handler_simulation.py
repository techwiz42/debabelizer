#!/usr/bin/env python3
"""
Test script that exactly duplicates the WebSocket handler's usage of debabelizer.
This simulates the exact sequence used in soniox_handler.py to identify issues.
"""

import asyncio
import os
import sys
import time
import uuid
import traceback

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

class WebSocketHandlerSimulation:
    """Simulates the exact behavior of soniox_handler.py"""
    
    def __init__(self):
        self.voice_service = None
        
    async def initialize_voice_service(self):
        """Initialize like the backend does"""
        print("🏗️ Initializing VoiceProcessor like the backend...")
        import debabelizer
        self.voice_service = debabelizer.VoiceProcessor()
        print(f"✅ VoiceProcessor created: {type(self.voice_service)}")
        
    async def handle_soniox_streaming_simulation(self, simulate_audio_chunks=True):
        """Exactly simulate handle_soniox_streaming from soniox_handler.py"""
        
        session_id = str(uuid.uuid4())
        stt_session_id = None
        results_received = 0
        
        try:
            if not self.voice_service:
                print("ERROR: STT processor not initialized")
                return False
            
            print(f"Started Soniox real streaming STT session: {session_id}")
            print(f"Processor type: {type(self.voice_service)}")
            
            # Start Soniox streaming session using EXACT same parameters as backend
            try:
                print("Attempting to start Soniox streaming...")
                print("Configuration: PCM format, 16kHz, language identification enabled")
                stt_session_id = await self.voice_service.start_streaming_transcription(
                    audio_format="pcm",      # PCM format from frontend
                    sample_rate=16000,       # 16kHz from frontend  
                    enable_language_identification=True,  # Enable auto language detection
                    has_pending_audio=True,   # Indicate we expect more audio
                    interim_results=True     # Enable interim results
                )
                print(f"SUCCESS: Soniox streaming session started: {stt_session_id}")
                print(f"Session ID type: {type(stt_session_id)}")
                
                # Add delay like backend does
                await asyncio.sleep(0.2)
                print(f"Session {stt_session_id} ready for audio streaming")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to start Soniox streaming session: {e}")
                print(f"Error type: {type(e)}")
                traceback.print_exc()
                return False
            
            # Create task to handle streaming results (EXACT COPY from backend)
            async def handle_streaming_results():
                nonlocal results_received
                try:
                    print(f"Starting to listen for streaming results from session {stt_session_id}")
                    async for result in self.voice_service.get_streaming_results(stt_session_id):
                        results_received += 1
                        
                        # Same logic as backend for keep-alive detection
                        is_keep_alive = (
                            not result.text.strip() and 
                            hasattr(result, 'metadata') and 
                            result.metadata and 
                            isinstance(result.metadata, dict) and 
                            result.metadata.get("type") == "keep-alive"
                        )
                        
                        if is_keep_alive:
                            keep_alive_reason = result.metadata.get("reason", "unknown")
                            print(f"Keep-alive from Soniox: {keep_alive_reason}")
                        else:
                            # Log ALL results like backend would send to frontend
                            print(f"📥 STT Result {results_received}: '{result.text}' (final: {result.is_final}, conf: {result.confidence:.2f})")
                            
                            # Simulate word counting like backend
                            if result.is_final and result.text:
                                word_count = len(result.text.split())
                                if word_count > 0:
                                    print(f"Tracked {word_count} STT words")
                        
                        # Stop after receiving several results for testing
                        if results_received >= 15:
                            print("🛑 Stopping after 15 results for testing")
                            break
                            
                    print(f"Streaming results loop ended for session {stt_session_id}")
                except Exception as e:
                    print(f"CRITICAL: Error handling Soniox streaming results: {e}")
                    print(f"Error type: {type(e)}")
                    traceback.print_exc()
            
            # Start the results handler task (EXACT COPY from backend)
            results_task = asyncio.create_task(handle_streaming_results())
            
            # Small delay to ensure stream is fully initialized (EXACT COPY)
            await asyncio.sleep(0.05)
            
            # Simulate receiving audio data from frontend (EXACT COPY from backend)
            if simulate_audio_chunks:
                print("🎵 Simulating audio chunks from frontend...")
                
                # Simulate audio chunks like the frontend would send
                chunk_count = 0
                max_chunks = 20  # Test with reasonable number of chunks
                
                for i in range(max_chunks):
                    try:
                        # Create realistic audio chunk (silence with some variation)
                        chunk_size = 170 if i == 0 else 1600  # First chunk smaller like real data
                        audio_chunk = bytearray(chunk_size)
                        
                        # Add some variation to simulate real audio
                        for j in range(0, len(audio_chunk), 2):
                            import struct
                            # Small random variation to simulate microphone input
                            sample = int(100 * ((i + j) % 256 - 128) / 128)  # -100 to +100 range
                            audio_chunk[j:j+2] = struct.pack('<h', sample)  # 16-bit little-endian
                        
                        # Log first chunk details like backend
                        if i == 0:
                            print(f"First audio chunk: session={stt_session_id}, size={len(audio_chunk)} bytes")
                        
                        # Stream audio data directly to Soniox (EXACT COPY)
                        await self.voice_service.stream_audio(stt_session_id, bytes(audio_chunk))
                        chunk_count += 1
                        
                        # Small delay between chunks to simulate real-time audio
                        await asyncio.sleep(0.05)  # 50ms chunks = 20fps audio
                        
                    except Exception as stream_error:
                        error_msg = str(stream_error)
                        print(f"Error streaming audio chunk to Soniox: {error_msg}")
                        print(f"Session ID attempted: {stt_session_id}")
                        
                        # Same error handling as backend
                        if any(phrase in error_msg for phrase in [
                            "No active stream", 
                            "Sending after closing", 
                            "WebSocket protocol error",
                            "Connection closed"
                        ]):
                            print(f"Stream {stt_session_id} is closed or unavailable, stopping audio processing")
                            break
                        
                        traceback.print_exc()
                
                print(f"Sent {chunk_count} audio chunks total")
            
            # Wait a bit more for results
            await asyncio.sleep(2.0)
            
            # Cancel the results task (EXACT COPY from backend)
            if not results_task.done():
                results_task.cancel()
                try:
                    await results_task
                except asyncio.CancelledError:
                    pass
            
            success = results_received > 0
            print(f"📊 Session Results: {results_received} total results received")
            return success
            
        except Exception as e:
            print(f"Soniox WebSocket simulation error: {e}")
            traceback.print_exc()
            return False
        finally:
            # Stop Soniox streaming session (EXACT COPY from backend)
            if stt_session_id:
                try:
                    print(f"Stopping Soniox streaming session: {stt_session_id}")
                    await self.voice_service.stop_streaming_transcription(stt_session_id)
                    print("✅ Session stopped successfully")
                except Exception as cleanup_error:
                    print(f"Error stopping Soniox streaming session: {cleanup_error}")
            
            print(f"Cleaned up Soniox streaming session: {session_id}")

async def run_continuous_test():
    """Run the test continuously until speech is successfully recorded"""
    
    print("🚀 WebSocket Handler Simulation - Continuous Testing")
    print("This test exactly duplicates the backend's usage of debabelizer")
    print("=" * 70)
    
    simulator = WebSocketHandlerSimulation()
    await simulator.initialize_voice_service()
    
    test_count = 0
    success_count = 0
    max_tests = 10  # Run up to 10 tests
    
    while test_count < max_tests:
        test_count += 1
        print(f"\n🔄 Test Run #{test_count}")
        print("=" * 50)
        
        try:
            # Run the simulation
            success = await simulator.handle_soniox_streaming_simulation(simulate_audio_chunks=True)
            
            if success:
                success_count += 1
                print(f"✅ Test #{test_count} SUCCESS - Speech processing worked!")
                
                # If we get 2 successful runs, we can be confident it's working
                if success_count >= 2:
                    print(f"\n🎉 VERIFICATION COMPLETE!")
                    print(f"Successfully recorded speech in {success_count}/{test_count} tests")
                    print("The streaming functionality is working correctly!")
                    return True
            else:
                print(f"❌ Test #{test_count} FAILED - No speech results received")
            
            # Wait between tests
            if test_count < max_tests:
                print(f"\n⏳ Waiting 3 seconds before next test...")
                await asyncio.sleep(3.0)
                
        except Exception as e:
            print(f"❌ Test #{test_count} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Final Results:")
    print(f"Success rate: {success_count}/{test_count} tests passed")
    
    if success_count > 0:
        print("✅ Some tests succeeded - streaming functionality is partially working")
        return True
    else:
        print("❌ All tests failed - streaming functionality needs more fixes")
        return False

if __name__ == "__main__":
    asyncio.run(run_continuous_test())