#!/usr/bin/env python3
"""
Real-time streaming conversation example using Debabelizer

This example shows how to create a real-time voice conversation system
that listens to audio input and responds with synthesized speech.
"""

import asyncio
import sys
from pathlib import Path
import json
import time

# Add the src directory to the path so we can import debabelizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig


class ConversationManager:
    """Manages a streaming voice conversation"""
    
    def __init__(self, stt_provider: str = "soniox", tts_provider: str = "elevenlabs"):
        self.config = DebabelizerConfig()
        self.processor = VoiceProcessor(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            config=self.config
        )
        self.conversation_history = []
        self.session_id = None
        
    async def start_conversation(self):
        """Start a new conversation session"""
        
        print("🎤 Starting voice conversation...")
        print("🗣️  Speak into your microphone. Say 'exit' or 'quit' to end.")
        print("📱 Make sure your microphone is working and not muted.")
        print()
        
        # Start streaming STT session
        self.session_id = await self.processor.start_streaming_stt(
            language="en",  # Auto-detect language
            on_transcript=self._on_transcript_received,
            on_final=self._on_final_transcript
        )
        
        print(f"✅ Conversation session started (ID: {self.session_id})")
        print("🎤 Listening...")
        
        # Keep the session alive
        try:
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Conversation interrupted by user")
            await self.end_conversation()
            
    async def _on_transcript_received(self, transcript: str, is_final: bool):
        """Handle real-time transcript updates"""
        
        if not is_final:
            # Show partial transcripts
            print(f"\r🎤 Partial: {transcript}", end="", flush=True)
        else:
            # Clear the partial line and show final
            print(f"\r🎤 You said: {transcript}")
            
    async def _on_final_transcript(self, transcript: str, confidence: float):
        """Handle final transcript and generate response"""
        
        print(f"✅ Final transcript (confidence: {confidence:.2f}): {transcript}")
        
        # Check for exit commands
        if transcript.lower().strip() in ["exit", "quit", "goodbye", "stop"]:
            print("👋 Ending conversation...")
            await self.end_conversation()
            return
            
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": transcript,
            "timestamp": time.time()
        })
        
        # Generate a response (simple echo with processing)
        response = await self._generate_response(transcript)
        
        if response:
            # Add response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": time.time()
            })
            
            print(f"🤖 Assistant: {response}")
            
            # Synthesize and play response
            await self._speak_response(response)
            
    async def _generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input
        
        In a real application, this would call an LLM or chatbot API.
        For this example, we'll create simple rule-based responses.
        """
        
        user_input_lower = user_input.lower().strip()
        
        # Simple response patterns
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return "Hello! How can I help you today?"
            
        elif "how are you" in user_input_lower:
            return "I'm doing well, thank you for asking! How are you?"
            
        elif "what" in user_input_lower and "time" in user_input_lower:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."
            
        elif "weather" in user_input_lower:
            return "I don't have access to current weather data, but I hope it's nice outside!"
            
        elif "thank" in user_input_lower:
            return "You're welcome! Is there anything else I can help you with?"
            
        elif len(user_input.split()) == 1:
            return f"You said '{user_input}'. Could you tell me more about that?"
            
        else:
            return f"That's interesting! You mentioned: {user_input}. Can you elaborate on that?"
            
    async def _speak_response(self, response_text: str):
        """Synthesize and play the response"""
        
        try:
            print("🗣️  Generating speech...")
            
            # Create a temporary file for the response
            output_file = f"/tmp/response_{int(time.time())}.wav"
            
            result = await self.processor.synthesize_text(
                text=response_text,
                output_file=output_file
            )
            
            print(f"🎵 Speech generated ({result.duration:.1f}s)")
            
            # In a real application, you would play the audio file
            # For this example, we'll just indicate that speech was generated
            print("🔊 [Audio would play here - response synthesized successfully]")
            
            # Clean up temp file
            Path(output_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ Failed to generate speech: {e}")
            
    async def end_conversation(self):
        """End the conversation session"""
        
        if self.session_id:
            await self.processor.end_streaming_session(self.session_id)
            print(f"🛑 Ended conversation session: {self.session_id}")
            
        # Show conversation summary
        if self.conversation_history:
            print("\n" + "="*50)
            print("📜 CONVERSATION SUMMARY")
            print("="*50)
            
            for i, msg in enumerate(self.conversation_history, 1):
                role = "You" if msg["role"] == "user" else "Assistant"
                print(f"{i:2d}. {role:9}: {msg['content']}")
                
            print(f"\n💬 Total exchanges: {len(self.conversation_history)}")
            
        sys.exit(0)


async def test_providers():
    """Test which providers are available"""
    
    config = DebabelizerConfig()
    providers = config.get_configured_providers()
    
    print("🔍 Checking available providers...")
    
    stt_providers = providers["stt"]
    tts_providers = providers["tts"]
    
    if not stt_providers:
        print("❌ No STT providers configured!")
        print("Please configure at least one STT provider (Soniox, Deepgram, etc.)")
        return False
        
    if not tts_providers:
        print("❌ No TTS providers configured!")
        print("Please configure at least one TTS provider (ElevenLabs, etc.)")
        return False
        
    print(f"✅ STT providers available: {', '.join(stt_providers)}")
    print(f"✅ TTS providers available: {', '.join(tts_providers)}")
    
    return True


def main():
    """Main function"""
    
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        # Test provider availability
        asyncio.run(test_providers())
        return
    
    print("🎯 Debabelizer Streaming Conversation Example")
    print("=" * 50)
    
    # Check provider availability first
    if not asyncio.run(test_providers()):
        print("\n❌ Cannot start conversation - providers not configured")
        sys.exit(1)
    
    # Get provider preferences
    stt_provider = "soniox"  # Default
    tts_provider = "elevenlabs"  # Default
    
    if len(sys.argv) >= 2:
        stt_provider = sys.argv[1]
    if len(sys.argv) >= 3:
        tts_provider = sys.argv[2]
        
    print(f"\n🎤 STT Provider: {stt_provider}")
    print(f"🗣️  TTS Provider: {tts_provider}")
    print()
    
    # Start conversation
    conversation = ConversationManager(
        stt_provider=stt_provider,
        tts_provider=tts_provider
    )
    
    asyncio.run(conversation.start_conversation())


if __name__ == "__main__":
    main()