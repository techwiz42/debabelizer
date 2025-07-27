# Debabelizer Examples

This directory contains example scripts demonstrating how to use the Debabelizer voice processing library for various use cases.

## Prerequisites

Before running these examples, make sure you have:

1. **Installed Debabelizer** (run from project root):
   ```bash
   pip install -e .
   ```

2. **Configured API Keys** for your chosen providers:
   ```bash
   # For Soniox STT
   export SONIOX_API_KEY="your_soniox_key_here"
   
   # For Deepgram STT  
   export DEEPGRAM_API_KEY="your_deepgram_key_here"
   
   # For ElevenLabs TTS
   export ELEVENLABS_API_KEY="your_elevenlabs_key_here"
   
   # For OpenAI (Whisper STT)
   export OPENAI_API_KEY="your_openai_key_here"
   
   # For Azure Speech Services
   export AZURE_SPEECH_KEY="your_azure_key_here"
   export AZURE_SPEECH_REGION="your_azure_region"
   ```

3. **Audio Files** (for transcription examples):
   - WAV, MP3, FLAC, or other supported formats
   - Sample files can be downloaded or recorded

## Examples Overview

### 1. Basic Transcription (`basic_transcription.py`)

**Purpose**: Transcribe audio files using different STT providers

**Features**:
- Single file transcription with any configured STT provider
- Provider comparison mode to test multiple STT services
- Detailed results with confidence scores and language detection

**Usage**:
```bash
# Transcribe with default provider (Soniox)
python examples/basic_transcription.py sample.wav

# Transcribe with specific provider
python examples/basic_transcription.py sample.wav deepgram
python examples/basic_transcription.py sample.wav soniox

# Compare multiple providers
python examples/basic_transcription.py sample.wav compare
```

**Example Output**:
```
🎤 Transcribing sample.wav using soniox...
✅ Transcription complete!
📝 Text: Hello, this is a test recording for speech recognition.
🔍 Confidence: 0.95
⏱️  Duration: 3.2s
🌍 Language: en
📊 Word count: 9
```

### 2. Text-to-Speech (`text_to_speech.py`)

**Purpose**: Convert text to speech using different TTS providers

**Features**:
- Generate speech from text with any configured TTS provider
- List available voices for each provider
- Provider comparison mode to test different TTS services
- Custom voice selection support

**Usage**:
```bash
# Basic text synthesis
python examples/text_to_speech.py "Hello world!" output.wav

# With specific provider and voice
python examples/text_to_speech.py "Hello world!" output.wav elevenlabs 21m00Tcm4TlvDq8ikWAM

# List available voices
python examples/text_to_speech.py voices elevenlabs

# Compare providers
python examples/text_to_speech.py compare "Hello world!"
```

**Example Output**:
```
🗣️  Synthesizing text using elevenlabs...
📝 Text: Hello world!
✅ Synthesis complete!
🎵 Audio saved to: output.wav
📊 Audio format: wav
⏱️  Duration: 1.2s
📏 Size: 38400 bytes
🎭 Voice: Rachel (premium)
```

### 3. Streaming Conversation (`streaming_conversation.py`)

**Purpose**: Real-time voice conversation system with live STT and TTS

**Features**:
- Real-time speech recognition from microphone
- Automatic response generation
- Text-to-speech output for responses
- Conversation history tracking
- Natural conversation flow

**Usage**:
```bash
# Start conversation with default providers
python examples/streaming_conversation.py

# With specific providers
python examples/streaming_conversation.py soniox elevenlabs

# Test provider availability
python examples/streaming_conversation.py test
```

**Example Output**:
```
🎤 Starting voice conversation...
🗣️  Speak into your microphone. Say 'exit' or 'quit' to end.
✅ Conversation session started (ID: conv_1234567890)
🎤 Listening...

🎤 You said: Hello, how are you today?
✅ Final transcript (confidence: 0.92): Hello, how are you today?
🤖 Assistant: Hello! I'm doing well, thank you for asking! How are you?
🗣️  Generating speech...
🎵 Speech generated (2.1s)
🔊 [Audio would play here - response synthesized successfully]
```

### 4. Telephony Integration (`telephony_integration.py`)

**Purpose**: Telephony/call center integration with real-time voice processing

**Features**:
- Simulates incoming call handling
- Real-time transcription of caller speech
- Automated customer service responses
- Telephony audio format support (mulaw, 8kHz)
- Call transcript generation and logging
- Multi-call simulation for call centers

**Usage**:
```bash
# Handle single call
python examples/telephony_integration.py CALL-001

# Simulate multiple concurrent calls
python examples/telephony_integration.py multi
```

**Example Output**:
```
📞 Incoming call: CALL-001
🔄 Setting up voice processing...
✅ Call session started (ID: tel_1234567890)
🗣️  Playing greeting: Hello! Thank you for calling. How can I assist you today?
🎵 Greeting generated (3.5s, mulaw format)
📞 [Greeting audio would be streamed to caller]

📞 Caller: I need help with my billing
✅ Final transcript (confidence: 0.89): I need help with my billing
🤖 System response: I can help you with billing questions. Let me pull up your account information...
🎵 Response generated (4.2s, mulaw format)
📞 [Response audio would be streamed to caller]
```

## Advanced Usage

### Provider Configuration

Each example automatically detects configured providers. You can check which providers are available:

```python
from debabelizer import DebabelizerConfig

config = DebabelizerConfig()
providers = config.get_configured_providers()

print("STT Providers:", providers["stt"])
print("TTS Providers:", providers["tts"])
```

### Custom Configuration

You can customize provider settings:

```python
from debabelizer import VoiceProcessor, DebabelizerConfig

# Custom config
config = DebabelizerConfig({
    "soniox": {
        "model": "stt-rt-preview",
        "enable_profanity_filter": True,
    },
    "elevenlabs": {
        "model": "eleven_turbo_v2_5",
        "optimize_streaming_latency": 1,
    }
})

processor = VoiceProcessor(
    stt_provider="soniox",
    tts_provider="elevenlabs", 
    config=config,
    optimize_for="quality"  # or "latency", "cost"
)
```

### Error Handling

All examples include comprehensive error handling:

```python
try:
    result = await processor.transcribe_file("audio.wav")
    print(f"Transcription: {result.text}")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Audio Format Support

Debabelizer supports various audio formats:

- **Input**: WAV, MP3, FLAC, OGG, M4A, AAC, WebM
- **Streaming**: PCM, mulaw (telephony)
- **Output**: WAV, MP3, mulaw (provider-dependent)

## Performance Tips

1. **Optimize for Use Case**:
   - `optimize_for="latency"` for real-time applications
   - `optimize_for="quality"` for high-quality results
   - `optimize_for="cost"` for cost-effective processing

2. **Provider Selection**:
   - **Soniox**: Best for accuracy and language detection
   - **Deepgram**: Fast and reliable, good for streaming
   - **ElevenLabs**: Highest quality TTS voices
   - **OpenAI**: Good general-purpose option

3. **Audio Quality**:
   - Use 16kHz sample rate for best results
   - Ensure good microphone quality for streaming
   - Use appropriate formats for your use case

## Troubleshooting

### Common Issues

1. **"Provider not configured"**:
   - Check that API keys are set as environment variables
   - Verify the API key is valid and has sufficient credits

2. **"FFmpeg not available"**:
   - Install FFmpeg: `apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)
   - Required for audio format conversion

3. **Streaming issues**:
   - Check microphone permissions
   - Ensure audio input device is working
   - Verify network connectivity for streaming APIs

4. **Audio quality issues**:
   - Use appropriate sample rates (8kHz for telephony, 16kHz+ for quality)
   - Check for background noise
   - Ensure proper audio levels

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration Examples

These examples demonstrate integration patterns for:

- **Web Applications**: Real-time voice features
- **Mobile Apps**: Voice commands and responses  
- **Call Centers**: Automated customer service
- **Voice Assistants**: Conversational AI systems
- **Accessibility Tools**: Speech-to-text and text-to-speech
- **Content Creation**: Voice-over generation

## Next Steps

After trying these examples:

1. **Extend the Examples**: Add your own response logic, integrate with APIs
2. **Create Custom Providers**: Implement additional STT/TTS services
3. **Build Applications**: Use Debabelizer in your own projects
4. **Optimize Performance**: Fine-tune for your specific use case
5. **Add Features**: Language translation, sentiment analysis, etc.

## Contributing

Found issues or want to add more examples? Contributions are welcome!

- Report bugs or suggest features
- Add new provider integrations
- Create additional example use cases
- Improve documentation and examples