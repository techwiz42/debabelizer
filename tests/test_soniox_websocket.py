#!/usr/bin/env python3
"""
Test script to debug Soniox WebSocket connection issues
"""

import asyncio
import json
import websockets
import sys

async def test_soniox_connection():
    api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"
    ws_url = "wss://stt-rt.soniox.com/transcribe-websocket"
    
    print(f"Connecting to: {ws_url}")
    
    try:
        # Connect with Bearer auth header
        headers = {"Authorization": f"Bearer {api_key}"}
        websocket = await websockets.connect(
            ws_url,
            additional_headers=headers,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=30
        )
        print("✓ WebSocket connected")
        
        # Send configuration
        config = {
            "api_key": api_key,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1,
            "model": "stt-rt-preview-v2",
            "enable_language_identification": True,
            "include_nonfinal": True
        }
        
        print(f"Sending config: {json.dumps(config, indent=2)}")
        await websocket.send(json.dumps(config))
        print("✓ Configuration sent")
        
        # Try to receive messages
        print("\nWaiting for messages...")
        message_count = 0
        
        try:
            while True:
                # Wait for a message with timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message_count += 1
                print(f"\nMessage {message_count}:")
                print(f"Raw: {message[:200]}..." if len(message) > 200 else f"Raw: {message}")
                
                try:
                    data = json.loads(message)
                    print(f"Parsed: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError:
                    print("(Not JSON)")
                    
        except asyncio.TimeoutError:
            print("\nNo messages received in 5 seconds")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"\nConnection closed: code={e.code}, reason={e.reason}")
        except Exception as e:
            print(f"\nError receiving messages: {type(e).__name__}: {e}")
            
        # Check if websocket is still open
        try:
            websocket.ping()
            websocket_open = True
        except:
            websocket_open = False
            
        if websocket_open:
            print("\nWebSocket is still open, sending test audio...")
            
            # Send a small test audio chunk (silence)
            test_audio = b'\x00' * 320  # 10ms of silence at 16kHz
            await websocket.send(test_audio)
            print("✓ Test audio sent")
            
            # Wait for response
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"\nReceived after audio: {message}")
            except asyncio.TimeoutError:
                print("\nNo response to audio within 2 seconds")
            except Exception as e:
                print(f"\nError after sending audio: {type(e).__name__}: {e}")
        else:
            print("\nWebSocket is closed")
            
        # Close connection
        await websocket.close()
        print("\n✓ Connection closed gracefully")
        
    except Exception as e:
        print(f"\n✗ Connection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Soniox WebSocket Connection Test")
    print("=" * 40)
    asyncio.run(test_soniox_connection())