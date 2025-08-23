#!/usr/bin/env python3
"""
Detailed test to show exactly when and why the Soniox WebSocket is closing.
This will help identify if it's a server-side rejection or a client-side issue.
"""

import asyncio
import os
import sys
import traceback
import time
import websockets
import json
from typing import AsyncGenerator

# Add the Python package path  
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_raw_soniox_websocket():
    """Test the raw WebSocket connection to Soniox to see what happens."""
    
    print("🔍 Testing raw Soniox WebSocket connection...")
    print("=" * 60)
    
    api_key = os.getenv('SONIOX_API_KEY')
    if not api_key:
        print("❌ SONIOX_API_KEY not found")
        return
        
    print(f"✅ API Key: {api_key[:8]}...")
    
    websocket_url = "wss://stt-rt.soniox.com/transcribe-websocket"
    print(f"🌐 Connecting to: {websocket_url}")
    
    try:
        # Connect with proper headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Host": "stt-rt.soniox.com"
        }
        
        print("📡 Establishing WebSocket connection...")
        start_time = time.time()
        
        async with websockets.connect(websocket_url, extra_headers=headers) as websocket:
            connect_time = time.time() - start_time
            print(f"✅ Connected in {connect_time:.2f}s")
            
            # Send configuration exactly as the Rust implementation does
            config = {
                "api_key": api_key,
                "audio_format": "pcm_s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "model": "stt-rt-preview-v2",
                "enable_language_identification": True,
                "include_nonfinal": True,
            }
            
            print("📤 Sending configuration...")
            config_start = time.time()
            await websocket.send(json.dumps(config))
            
            # Wait for initial response
            print("⏳ Waiting for initial response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_time = time.time() - config_start
                print(f"📥 Got response in {response_time:.2f}s: {response}")
                
                # Parse the response
                try:
                    resp_data = json.loads(response)
                    print(f"✅ Parsed response: {json.dumps(resp_data, indent=2)}")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse response as JSON: {e}")
                
            except asyncio.TimeoutError:
                print("⏱️ No response within 5 seconds")
                
            # Try to send some dummy audio
            print("🎵 Sending dummy audio...")
            dummy_audio = b'\x00' * 1600  # 0.05 seconds of silence
            
            try:
                await websocket.send(dummy_audio)
                print("✅ Sent dummy audio")
                
                # Wait a bit and listen for more responses
                print("👂 Listening for more responses...")
                for i in range(3):
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"📥 Response {i+1}: {response}")
                    except asyncio.TimeoutError:
                        print(f"⏱️ No response {i+1} after 2s")
                        break
                        
            except Exception as audio_error:
                print(f"❌ Error sending audio: {audio_error}")
                
            print("🔚 Test completed - connection should still be alive")
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket closed: {e.code} - {e.reason}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        traceback.print_exc()

async def test_debabelizer_with_logging():
    """Test debabelizer but with enhanced logging to see what's happening."""
    
    print("\n" + "=" * 60)
    print("🔍 Testing debabelizer with enhanced logging...")
    print("=" * 60)
    
    try:
        import debabelizer
        
        # Enable more verbose logging if possible
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        print("🏗️ Creating VoiceProcessor...")
        processor = debabelizer.VoiceProcessor()
        
        print("🚀 Starting streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        print(f"✅ Session: {session_id}")
        
        # Immediately try to get results
        print("📥 Getting streaming results iterator...")
        results_iter = processor.get_streaming_results(session_id)
        
        print("🔄 Testing iterator immediately (no delay)...")
        result_count = 0
        start_time = time.time()
        
        try:
            async for result in results_iter:
                result_count += 1
                elapsed = time.time() - start_time
                print(f"   Result {result_count} (t={elapsed:.2f}s): '{result.text}' final={result.is_final}")
                
                # Break after a few results
                if result_count >= 5:
                    break
                    
            print(f"🏁 Iterator ended after {result_count} results")
            
        except Exception as iter_error:
            elapsed = time.time() - start_time
            print(f"❌ Iterator failed after {elapsed:.2f}s: {iter_error}")
            
    except Exception as e:
        print(f"❌ Debabelizer test failed: {e}")
        traceback.print_exc()

async def main():
    """Run both tests."""
    await test_raw_soniox_websocket()
    await test_debabelizer_with_logging()

if __name__ == "__main__":
    asyncio.run(main())