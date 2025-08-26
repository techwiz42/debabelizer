#!/usr/bin/env python3
"""Test Soniox WebSocket directly to see if it's working."""

import asyncio
import websockets
import json
import wave

async def test_soniox_websocket():
    """Test Soniox WebSocket directly."""
    api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"
    url = f"wss://stt-rt.soniox.com/transcribe-websocket?api_key={api_key}"
    
    print(f"Connecting to {url}")
    
    async with websockets.connect(url) as websocket:
        # Send initial configuration
        config = {
            "api_key": api_key,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1,
            "model": "stt-rt-preview-v2",
            "enable_language_identification": True,
            "include_nonfinal": True,
        }
        
        print(f"Sending config: {json.dumps(config)}")
        await websocket.send(json.dumps(config))
        
        # Wait for handshake
        response = await websocket.recv()
        print(f"Handshake response: {response}")
        
        # Load and send audio
        with wave.open("english_sample.wav", 'rb') as wav:
            audio_data = wav.readframes(wav.getnframes())
        
        print(f"Sending {len(audio_data)} bytes of audio")
        await websocket.send(audio_data)
        
        # Send end-of-audio signal
        print("Sending end-of-audio signal (empty bytes)")
        await websocket.send(b'')
        
        # Wait for responses
        print("\nWaiting for transcription results...")
        for i in range(20):  # Wait for up to 20 messages
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"\nMessage {i+1}: {response}")
                
                # Parse and check for actual transcription
                data = json.loads(response)
                if data.get('tokens'):
                    print("🎉 Got transcription!")
                    for token in data['tokens']:
                        print(f"  Token: {token}")
            except asyncio.TimeoutError:
                print(f"Timeout after {i} messages")
                break
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                break

if __name__ == "__main__":
    asyncio.run(test_soniox_websocket())