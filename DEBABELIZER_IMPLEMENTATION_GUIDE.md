# Debabelizer Migration Implementation Guide

## Executive Summary

This document provides a comprehensive implementation strategy to address the migration difficulties outlined in `DEBABELIZER_MIGRATION_ANALYSIS.md`. The primary challenge is that the new Rust-based debabelizer implementation with Python bindings does not provide the same streaming STT API that the application currently relies on.

**Key Finding**: The new debabelizer **DOES** support streaming STT via the `transcribe_stream()` method, but the API interface has completely changed.

## Root Cause Resolution

### Issue: "New API doesn't support streaming"

**Status**: ❌ **INCORRECT ASSUMPTION**

The migration analysis incorrectly concluded that streaming wasn't supported. Based on the Rust implementation review:

**Evidence of Streaming Support**:
- ✅ `SttStream` trait with `send_audio()`, `receive_transcript()`, `close()` methods
- ✅ `StreamConfig` for configuring streaming sessions
- ✅ `StreamingResult` for real-time results with interim/final flags
- ✅ `transcribe_stream()` method in `VoiceProcessor`
- ✅ WebSocket implementations for Soniox, Deepgram, Azure
- ✅ Working streaming example: `examples/streaming_stt.rs`

**New Streaming API Pattern**:
```python
# NEW STREAMING API
from debabelizer import VoiceProcessor, StreamConfig, AudioFormat

# Create processor
processor = VoiceProcessor(config=config_dict)

# Configure streaming
stream_config = StreamConfig(
    format=AudioFormat.wav(16000),
    interim_results=True,
    punctuate=True,
    language=None  # Auto-detect
)

# Create stream
stream = await processor.transcribe_stream(stream_config)

# Send audio chunks
await stream.send_audio(audio_chunk)

# Receive results
result = await stream.receive_transcript()
if result:
    print(f"[{'FINAL' if result.is_final else 'INTERIM'}] {result.text}")

# Close stream
await stream.close()
```

## Implementation Strategy

### Phase 1: Update WebSocket Handlers for New API ✅ **PRIORITY 1**

The core issue is that WebSocket handlers use non-existent methods from the old API. Here's the migration mapping:

#### API Migration Mapping

| Old API Method | New API Equivalent | Implementation |
|----------------|-------------------|----------------|
| `start_streaming_transcription()` | `transcribe_stream(StreamConfig)` | Create StreamConfig and call transcribe_stream |
| `stream_audio(session_id, chunk)` | `stream.send_audio(chunk)` | Send to SttStream instance |
| `get_streaming_results(session_id)` | `stream.receive_transcript()` | Receive from SttStream instance |
| `stop_streaming_transcription()` | `stream.close()` | Close SttStream instance |

#### Updated WebSocket Handler Pattern

```python
# NEW WEBSOCKET HANDLER PATTERN
from debabelizer import VoiceProcessor, StreamConfig, AudioFormat
import asyncio

async def handle_streaming_stt(websocket: WebSocket, current_user=None):
    await websocket.accept()
    stream = None
    
    try:
        # Initialize processor
        processor = VoiceProcessor(config=voice_service.config)
        
        # Configure streaming
        stream_config = StreamConfig(
            format=AudioFormat("wav", 16000, 1, 16),
            interim_results=True,
            punctuate=True,
            language=None
        )
        
        # Create streaming session
        stream = await processor.transcribe_stream(stream_config)
        
        # Start result processing task
        async def process_results():
            try:
                while True:
                    result = await stream.receive_transcript()
                    if result is None:
                        break
                    
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result.text,
                        "is_final": result.is_final,
                        "confidence": result.confidence,
                        "session_id": str(result.session_id),
                        "timestamp": result.timestamp.isoformat()
                    })
            except Exception as e:
                await websocket.send_json({"error": f"Transcription error: {e}"})
        
        # Start background task for results
        result_task = asyncio.create_task(process_results())
        
        # Handle incoming audio
        async for message in websocket.iter_bytes():
            await stream.send_audio(message)
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        if stream:
            await stream.close()
        if 'result_task' in locals():
            result_task.cancel()
```

### Phase 2: Update Voice Service Configuration ✅ **PRIORITY 1**

#### Current Issues in voice_service.py:
1. Import errors for removed classes
2. Configuration format mismatch
3. Method signature changes

#### Updated Voice Service Pattern:

```python
# UPDATED VOICE SERVICE
from debabelizer import VoiceProcessor, AudioFormat, AudioData
from debabelizer.utils import create_synthesis_options
import os

class VoiceService:
    def __init__(self):
        self.processor = None
        self.config = self._build_config()
    
    def _build_config(self):
        """Build configuration dictionary for new API"""
        config = {
            "preferences": {
                "stt_provider": os.getenv("DEBABELIZER_STT_PROVIDER", "deepgram"),
                "tts_provider": os.getenv("DEBABELIZER_TTS_PROVIDER", "openai"),
                "optimize_for": "quality"
            }
        }
        
        # Add provider configurations
        if os.getenv("DEEPGRAM_API_KEY"):
            config["deepgram"] = {"api_key": os.getenv("DEEPGRAM_API_KEY")}
        
        if os.getenv("OPENAI_API_KEY"):
            config["openai"] = {"api_key": os.getenv("OPENAI_API_KEY")}
        
        if os.getenv("SONIOX_API_KEY"):
            config["soniox"] = {"api_key": os.getenv("SONIOX_API_KEY")}
            
        return config
    
    async def initialize_processors(self):
        """Initialize voice processor with new API"""
        try:
            self.processor = VoiceProcessor(config=self.config)
            print("Voice processor initialized successfully")
        except Exception as e:
            print(f"Failed to initialize voice processor: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav"):
        """Transcribe audio using new API"""
        if not self.processor:
            await self.initialize_processors()
        
        # Create audio objects
        format_obj = AudioFormat(
            format=audio_format,
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )
        audio_obj = AudioData(audio_data, format_obj)
        
        # Transcribe
        result = self.processor.transcribe(audio_obj)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language_detected,
            "duration": result.duration
        }
    
    async def synthesize_speech(self, text: str, voice: str = "alloy", format: str = "mp3"):
        """Synthesize speech using new API"""
        if not self.processor:
            await self.initialize_processors()
        
        options = create_synthesis_options(
            voice=voice,
            speed=1.0,
            format=format
        )
        
        result = self.processor.synthesize(text, options)
        return {
            "audio_data": result.audio_data,
            "format": result.format,
            "duration": result.duration
        }
```

### Phase 3: Provider-Specific Handler Updates ✅ **PRIORITY 2**

Each WebSocket handler needs to be updated for the new streaming API:

#### 1. Deepgram Handler Updates

```python
# UPDATED DEEPGRAM HANDLER
async def handle_deepgram_streaming(websocket: WebSocket, current_user=None):
    await websocket.accept()
    stream = None
    
    try:
        # Create processor with Deepgram preference
        config = {
            "preferences": {"stt_provider": "deepgram"},
            "deepgram": {"api_key": os.getenv("DEEPGRAM_API_KEY")}
        }
        processor = VoiceProcessor(config=config)
        
        # Configure streaming for Deepgram
        stream_config = StreamConfig(
            format=AudioFormat("wav", 16000, 1, 16),
            interim_results=True,
            punctuate=True,
            diarization=False,
            language="en-US"
        )
        
        # Create stream
        stream = await processor.transcribe_stream(stream_config)
        
        # Handle streaming as in updated pattern above
        # ... (rest of implementation)
```

#### 2. Soniox Handler Updates

```python
# UPDATED SONIOX HANDLER  
async def handle_soniox_streaming(websocket: WebSocket, current_user=None):
    await websocket.accept()
    stream = None
    
    try:
        # Create processor with Soniox preference
        config = {
            "preferences": {"stt_provider": "soniox"},
            "soniox": {"api_key": os.getenv("SONIOX_API_KEY")}
        }
        processor = VoiceProcessor(config=config)
        
        # Configure streaming for Soniox (word-level optimization)
        stream_config = StreamConfig(
            format=AudioFormat("wav", 16000, 1, 16),
            interim_results=True,
            punctuate=True,
            enable_word_time_offsets=True  # Soniox specialty
        )
        
        # Create stream
        stream = await processor.transcribe_stream(stream_config)
        
        # Soniox-specific word-level processing
        current_words = []
        
        async def process_soniox_results():
            async for result in stream.receive_transcript():
                if result.words:
                    # Update word-level display
                    for word in result.words:
                        current_words.append({
                            "word": word.word,
                            "confidence": word.confidence,
                            "start": word.start,
                            "end": word.end
                        })
                
                await websocket.send_json({
                    "type": "transcription",
                    "text": result.text,
                    "is_final": result.is_final,
                    "confidence": result.confidence,
                    "words": current_words if result.is_final else None,
                    "session_id": str(result.session_id)
                })
                
                if result.is_final:
                    current_words = []
        
        # ... (rest of implementation)
```

#### 3. Whisper Handler Updates

```python
# UPDATED WHISPER HANDLER
async def handle_whisper_transcription(websocket: WebSocket, current_user=None):
    """
    Note: Whisper in new API is batch-only, so this simulates streaming
    by buffering audio and processing in chunks
    """
    await websocket.accept()
    
    try:
        # Create processor with Whisper preference
        config = {
            "preferences": {"stt_provider": "whisper"},
            "whisper": {
                "model_size": "base",
                "device": "auto",
                "language": "en"
            }
        }
        processor = VoiceProcessor(config=config)
        
        # Buffer for accumulating audio
        audio_buffer = bytearray()
        buffer_duration = 2.0  # Process every 2 seconds
        sample_rate = 16000
        chunk_size = int(sample_rate * buffer_duration * 2)  # 16-bit audio
        
        async for message in websocket.iter_bytes():
            audio_buffer.extend(message)
            
            # Process when buffer is full
            if len(audio_buffer) >= chunk_size:
                # Extract chunk
                chunk = bytes(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]
                
                # Transcribe chunk
                audio_obj = AudioData(
                    chunk, 
                    AudioFormat("wav", sample_rate, 1, 16)
                )
                result = processor.transcribe(audio_obj)
                
                if result.text.strip():
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result.text,
                        "is_final": True,  # Whisper only provides final results
                        "confidence": result.confidence,
                        "language": result.language_detected
                    })
    
    except Exception as e:
        await websocket.send_json({"error": str(e)})
```

### Phase 4: Frontend Compatibility ✅ **PRIORITY 2**

The frontend WebSocket client should largely work without changes, but some adjustments may be needed:

#### Frontend WebSocket Updates:

```typescript
// FRONTEND WEBSOCKET CLIENT UPDATES
interface TranscriptionResult {
  type: 'transcription';
  text: string;
  is_final: boolean;
  confidence: number;
  session_id: string;
  timestamp?: string;
  words?: Array<{
    word: string;
    confidence: number;
    start: number;
    end: number;
  }>;
}

class STTWebSocketClient {
  private ws: WebSocket | null = null;
  
  connect(sessionToken: string) {
    const wsUrl = `ws://localhost:8000/ws/stt?session_token=${sessionToken}`;
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'transcription') {
        this.handleTranscription(data as TranscriptionResult);
      } else if (data.error) {
        this.handleError(data.error);
      }
    };
  }
  
  sendAudio(audioData: ArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }
  
  private handleTranscription(result: TranscriptionResult) {
    // Handle interim vs final results
    if (result.is_final) {
      // Update final transcript
      this.onFinalTranscript(result.text, result.confidence);
    } else {
      // Update interim display
      this.onInterimTranscript(result.text, result.confidence);
    }
    
    // Handle word-level updates for Soniox
    if (result.words) {
      this.onWordsUpdate(result.words);
    }
  }
}
```

### Phase 5: Configuration Management ✅ **PRIORITY 3**

#### Environment Variable Updates:

```bash
# NEW ENVIRONMENT VARIABLES
DEBABELIZER_STT_PROVIDER=deepgram
DEBABELIZER_TTS_PROVIDER=openai
DEBABELIZER_AUTO_SELECT=false
DEBABELIZER_OPTIMIZE_FOR=quality

# Provider API Keys (unchanged)
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
SONIOX_API_KEY=your_soniox_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Provider-specific settings
DEEPGRAM_MODEL=nova-2
OPENAI_TTS_MODEL=tts-1-hd
OPENAI_TTS_VOICE=alloy
```

#### Settings Class Updates:

```python
# UPDATED SETTINGS CLASS
class Settings:
    # New debabelizer configuration
    debabelizer_stt_provider: str = "deepgram"
    debabelizer_tts_provider: str = "openai"
    debabelizer_auto_select: bool = False
    debabelizer_optimize_for: str = "quality"
    
    # Provider API keys
    deepgram_api_key: str = ""
    openai_api_key: str = ""
    soniox_api_key: str = ""
    elevenlabs_api_key: str = ""
    
    def get_debabelizer_config(self) -> dict:
        """Build debabelizer configuration dictionary"""
        config = {
            "preferences": {
                "stt_provider": self.debabelizer_stt_provider,
                "tts_provider": self.debabelizer_tts_provider,
                "auto_select": self.debabelizer_auto_select,
                "optimize_for": self.debabelizer_optimize_for
            }
        }
        
        # Add provider configs
        if self.deepgram_api_key:
            config["deepgram"] = {"api_key": self.deepgram_api_key}
        if self.openai_api_key:
            config["openai"] = {"api_key": self.openai_api_key}
        if self.soniox_api_key:
            config["soniox"] = {"api_key": self.soniox_api_key}
        if self.elevenlabs_api_key:
            config["elevenlabs"] = {"api_key": self.elevenlabs_api_key}
            
        return config
```

## Implementation Timeline

### Week 1: Core API Migration ⚡ **CRITICAL**
- [ ] Update `voice_service.py` for new API patterns
- [ ] Update `main.py` endpoints for new object types
- [ ] Test basic STT/TTS functionality without streaming
- [ ] Verify backend can start without import errors

### Week 2: Streaming Implementation 🚀 **HIGH PRIORITY**
- [ ] Implement new streaming pattern in base WebSocket handler
- [ ] Update Deepgram handler for new streaming API
- [ ] Update Soniox handler for new streaming API  
- [ ] Update Whisper handler for buffered processing
- [ ] Test streaming functionality end-to-end

### Week 3: Optimization & Testing 🔧 **MEDIUM PRIORITY**
- [ ] Frontend WebSocket client adjustments
- [ ] Provider-specific optimizations
- [ ] Error handling improvements
- [ ] Performance testing vs old implementation
- [ ] Documentation updates

### Week 4: Production Readiness 🎯 **NICE TO HAVE**
- [ ] Monitoring and logging integration
- [ ] Fallback mechanisms between providers
- [ ] Load testing with concurrent connections
- [ ] Security review of new API usage

## Risk Mitigation

### High Risk: Streaming Performance
**Risk**: New streaming API may have different latency characteristics
**Mitigation**: 
- Benchmark latency vs old implementation
- Add configurable buffering options
- Implement fallback to shorter chunks if needed

### Medium Risk: Provider Feature Parity
**Risk**: New API may not expose all provider-specific features
**Mitigation**:
- Document feature mapping for each provider
- Test advanced features (diarization, word timing, etc.)
- Contact debabelizer team for missing features

### Low Risk: Frontend Compatibility
**Risk**: Frontend may need adjustments for new WebSocket message format
**Mitigation**:
- Maintain backward-compatible message format where possible
- Add version field to messages for future compatibility
- Test thoroughly with existing frontend

## Success Criteria

### ✅ Immediate Success (Week 1)
- [ ] Backend starts without import errors
- [ ] Basic STT endpoint works with file uploads
- [ ] Basic TTS endpoint works with text input
- [ ] Configuration loads correctly from environment

### ✅ Core Success (Week 2)  
- [ ] Real-time streaming STT works via WebSocket
- [ ] All three providers (Deepgram, Soniox, Whisper) function
- [ ] Interim and final results work correctly
- [ ] Frontend can connect and receive transcriptions

### ✅ Full Success (Week 3)
- [ ] Performance matches or exceeds old implementation
- [ ] Provider-specific features work (word timing, diarization)
- [ ] Error handling and reconnection work properly
- [ ] All existing frontend features functional

### ✅ Production Success (Week 4)
- [ ] Concurrent user sessions work reliably
- [ ] Monitoring and logging provide visibility
- [ ] Documentation complete for maintenance
- [ ] Team confident in production deployment

## Next Actions

### Immediate (Today)
1. **Install new debabelizer** in backend virtual environment
2. **Update voice_service.py imports** to use new API
3. **Test basic initialization** to confirm import resolution
4. **Start backend** to verify no more import errors

### This Week
1. **Implement base streaming pattern** in one WebSocket handler
2. **Test streaming with frontend** to verify message flow
3. **Update remaining WebSocket handlers** one by one
4. **Document any API differences** encountered

### Next Week  
1. **Performance benchmark** new vs old implementation
2. **Feature testing** for provider-specific capabilities
3. **Error handling improvements** and edge case testing
4. **Frontend optimization** for new message formats

This implementation guide provides a clear path forward to resolve all migration difficulties while maintaining full feature parity with the existing application.