#!/usr/bin/env python3
"""
Telephony integration example using Debabelizer

This example shows how to integrate Debabelizer with telephony systems
for real-time voice processing in call centers or voice bots.
"""

import asyncio
import sys
from pathlib import Path
import json
import time
from typing import Optional

# Add the src directory to the path so we can import debabelizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig


class TelephonyCallHandler:
    """Handles a single telephony call with voice processing"""
    
    def __init__(self, call_id: str, stt_provider: str = "soniox", tts_provider: str = "elevenlabs"):
        self.call_id = call_id
        self.config = DebabelizerConfig()
        self.processor = VoiceProcessor(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            config=self.config,
            optimize_for="latency"  # Optimize for low latency in telephony
        )
        self.session_id = None
        self.call_transcript = []
        self.call_start_time = time.time()
        
    async def handle_incoming_call(self):
        """Handle an incoming telephony call"""
        
        print(f"📞 Incoming call: {self.call_id}")
        print("🔄 Setting up voice processing...")
        
        # Start streaming STT for real-time transcription
        self.session_id = await self.processor.start_streaming_stt(
            language="auto",  # Auto-detect caller language
            sample_rate=8000,  # Telephony standard
            audio_format="mulaw",  # Telephony audio format
            on_transcript=self._on_realtime_transcript,
            on_final=self._on_final_transcript
        )
        
        print(f"✅ Call session started (ID: {self.session_id})")
        
        # Play greeting
        await self._play_greeting()
        
        # Keep call active
        try:
            while True:
                await asyncio.sleep(0.1)  # Process audio in real-time
                
        except KeyboardInterrupt:
            await self._end_call()
            
    async def _play_greeting(self):
        """Play initial greeting to caller"""
        
        greeting = "Hello! Thank you for calling. How can I assist you today?"
        
        print(f"🗣️  Playing greeting: {greeting}")
        
        try:
            # Generate greeting audio in telephony format
            result = await self.processor.synthesize_text(
                text=greeting,
                audio_format="mulaw",
                sample_rate=8000,
                output_file=f"/tmp/greeting_{self.call_id}.mulaw"
            )
            
            print(f"🎵 Greeting generated ({result.duration:.1f}s, mulaw format)")
            
            # In a real telephony system, this would be sent to the call stream
            print("📞 [Greeting audio would be streamed to caller]")
            
            # Add to transcript
            self.call_transcript.append({
                "speaker": "system",
                "text": greeting,
                "timestamp": time.time() - self.call_start_time,
                "type": "greeting"
            })
            
        except Exception as e:
            print(f"❌ Failed to generate greeting: {e}")
            
    async def _on_realtime_transcript(self, transcript: str, is_final: bool):
        """Handle real-time transcript updates from caller"""
        
        if not is_final:
            # Show partial transcripts for monitoring
            print(f"\r📞 Caller (partial): {transcript}", end="", flush=True)
        else:
            print(f"\r📞 Caller: {transcript}")
            
    async def _on_final_transcript(self, transcript: str, confidence: float):
        """Process final transcript and generate response"""
        
        timestamp = time.time() - self.call_start_time
        
        print(f"✅ Final transcript (confidence: {confidence:.2f}): {transcript}")
        
        # Add to call transcript
        self.call_transcript.append({
            "speaker": "caller",
            "text": transcript,
            "timestamp": timestamp,
            "confidence": confidence
        })
        
        # Check for call termination
        if any(word in transcript.lower() for word in ["goodbye", "bye", "hang up", "end call"]):
            await self._end_call()
            return
            
        # Generate response based on caller input
        response = await self._generate_call_response(transcript)
        
        if response:
            await self._speak_to_caller(response)
            
    async def _generate_call_response(self, caller_input: str) -> Optional[str]:
        """
        Generate appropriate response for telephony context
        
        In a real system, this would integrate with:
        - Customer service knowledge base
        - CRM system for customer data
        - Call routing logic
        - Business-specific workflows
        """
        
        caller_lower = caller_input.lower().strip()
        
        # Customer service response patterns
        if any(word in caller_lower for word in ["help", "support", "problem", "issue"]):
            return "I understand you need assistance. Let me connect you with our support team, or I can try to help you directly. Could you please describe the issue you're experiencing?"
            
        elif any(word in caller_lower for word in ["billing", "payment", "charge", "invoice"]):
            return "I can help you with billing questions. Let me pull up your account information. Could you please provide your account number or the phone number associated with your account?"
            
        elif any(word in caller_lower for word in ["hours", "open", "closed", "schedule"]):
            return "Our customer service hours are Monday through Friday, 9 AM to 6 PM Eastern Time. We're also available on weekends from 10 AM to 4 PM. Is there anything specific I can help you with today?"
            
        elif any(word in caller_lower for word in ["cancel", "cancellation", "terminate"]):
            return "I understand you're interested in cancellation. I'd like to help resolve any concerns you might have first. Could you tell me what's prompting you to consider canceling?"
            
        elif any(word in caller_lower for word in ["speak", "talk", "human", "agent", "representative"]):
            return "Of course! I'll connect you with one of our customer service representatives. Please hold for just a moment while I transfer your call."
            
        elif len(caller_input.split()) <= 3:
            return f"I heard you say '{caller_input}'. Could you provide a bit more detail about what you need help with?"
            
        else:
            return "Thank you for that information. Let me see how I can best assist you with that. One moment please."
            
    async def _speak_to_caller(self, response_text: str):
        """Generate and stream response to caller in telephony format"""
        
        print(f"🤖 System response: {response_text}")
        
        try:
            # Generate response in telephony format (mulaw, 8kHz)
            output_file = f"/tmp/response_{self.call_id}_{int(time.time())}.mulaw"
            
            result = await self.processor.synthesize_text(
                text=response_text,
                audio_format="mulaw",
                sample_rate=8000,
                output_file=output_file
            )
            
            print(f"🎵 Response generated ({result.duration:.1f}s, mulaw format)")
            print("📞 [Response audio would be streamed to caller]")
            
            # Add to transcript
            self.call_transcript.append({
                "speaker": "system",
                "text": response_text,
                "timestamp": time.time() - self.call_start_time,
                "audio_duration": result.duration
            })
            
            # Clean up temp file
            Path(output_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ Failed to generate call response: {e}")
            
    async def _end_call(self):
        """End the telephony call and cleanup"""
        
        call_duration = time.time() - self.call_start_time
        
        print(f"\n📞 Ending call {self.call_id}")
        print(f"⏱️  Call duration: {call_duration:.1f} seconds")
        
        if self.session_id:
            await self.processor.end_streaming_session(self.session_id)
            
        # Generate call summary
        await self._generate_call_summary()
        
        sys.exit(0)
        
    async def _generate_call_summary(self):
        """Generate a summary of the call"""
        
        print("\n" + "="*60)
        print(f"📋 CALL SUMMARY - {self.call_id}")
        print("="*60)
        
        if not self.call_transcript:
            print("No transcript available")
            return
            
        print(f"📞 Call Duration: {time.time() - self.call_start_time:.1f} seconds")
        print(f"💬 Total Exchanges: {len([t for t in self.call_transcript if t['speaker'] == 'caller'])}")
        print()
        
        print("📜 TRANSCRIPT:")
        print("-" * 40)
        
        for entry in self.call_transcript:
            timestamp = f"[{entry['timestamp']:.1f}s]"
            speaker = entry['speaker'].upper()
            text = entry['text']
            
            if entry['speaker'] == 'caller' and 'confidence' in entry:
                confidence = f" (conf: {entry['confidence']:.2f})"
            else:
                confidence = ""
                
            print(f"{timestamp:>8} {speaker:>7}: {text}{confidence}")
            
        # Save transcript to file
        transcript_file = f"call_transcript_{self.call_id}.json"
        with open(transcript_file, 'w') as f:
            json.dump({
                "call_id": self.call_id,
                "start_time": self.call_start_time,
                "duration": time.time() - self.call_start_time,
                "transcript": self.call_transcript
            }, f, indent=2)
            
        print(f"\n💾 Transcript saved to: {transcript_file}")


async def simulate_multiple_calls():
    """Simulate handling multiple concurrent calls"""
    
    print("🏢 Simulating telephony call center with multiple concurrent calls")
    print("=" * 60)
    
    # Create multiple call handlers
    call_handlers = [
        TelephonyCallHandler(f"CALL-{i+1:03d}")
        for i in range(3)  # Simulate 3 concurrent calls
    ]
    
    print(f"📞 Starting {len(call_handlers)} concurrent calls...")
    
    # Handle calls concurrently
    tasks = [
        asyncio.create_task(handler.handle_incoming_call())
        for handler in call_handlers
    ]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down call center...")
        for task in tasks:
            task.cancel()


def main():
    """Main function"""
    
    if len(sys.argv) >= 2 and sys.argv[1] == "multi":
        # Simulate multiple concurrent calls
        asyncio.run(simulate_multiple_calls())
        return
    
    print("📞 Debabelizer Telephony Integration Example")
    print("=" * 50)
    
    # Check configuration
    config = DebabelizerConfig()
    providers = config.get_configured_providers()
    
    if not providers["stt"] or not providers["tts"]:
        print("❌ STT and TTS providers must be configured for telephony")
        sys.exit(1)
        
    print(f"✅ STT: {', '.join(providers['stt'])}")
    print(f"✅ TTS: {', '.join(providers['tts'])}")
    
    # Get call ID
    call_id = sys.argv[1] if len(sys.argv) > 1 else f"CALL-{int(time.time())}"
    
    # Start call handler
    call_handler = TelephonyCallHandler(call_id)
    asyncio.run(call_handler.handle_incoming_call())


if __name__ == "__main__":
    main()