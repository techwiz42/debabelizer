use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyAny};
use pyo3::create_exception;
use tokio::runtime::Runtime;
use pyo3_asyncio::tokio::future_into_py;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

// Import the Rust debabelizer types
use debabelizer_core::{
    AudioData, AudioFormat as CoreAudioFormat, TranscriptionResult as CoreTranscriptionResult,
    SynthesisResult as CoreSynthesisResult, Voice as CoreVoice, WordTiming as CoreWordTiming,
    StreamingResult as CoreStreamingResult, SynthesisOptions as CoreSynthesisOptions, 
    SttStream,
};
use debabelizer::{DebabelizerConfig as CoreConfig, VoiceProcessor as CoreProcessor, DebabelizerError};

/// Python wrapper for AudioFormat
#[pyclass(name = "AudioFormat")]
#[derive(Clone)]
pub struct PyAudioFormat {
    pub inner: CoreAudioFormat,
}

#[pymethods]
impl PyAudioFormat {
    #[new]
    #[pyo3(signature = (format, sample_rate, channels=None, bit_depth=None))]
    fn new(format: String, sample_rate: u32, channels: Option<u8>, bit_depth: Option<u16>) -> Self {
        Self {
            inner: CoreAudioFormat {
                format,
                sample_rate,
                channels: channels.unwrap_or(1),
                bit_depth: Some(bit_depth.unwrap_or(16)),
            },
        }
    }

    #[getter]
    fn format(&self) -> String {
        self.inner.format.clone()
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> u8 {
        self.inner.channels
    }

    #[getter]
    fn bit_depth(&self) -> u16 {
        self.inner.bit_depth.unwrap_or(16)
    }
}

/// Python wrapper for AudioData
#[pyclass(name = "AudioData")]
#[derive(Clone)]
pub struct PyAudioData {
    pub inner: AudioData,
}

#[pymethods]
impl PyAudioData {
    #[new]
    fn new(data: Vec<u8>, format: PyAudioFormat) -> Self {
        Self {
            inner: AudioData {
                data,
                format: format.inner,
            },
        }
    }

    #[getter]
    fn data(&self) -> Vec<u8> {
        self.inner.data.clone()
    }

    #[getter]
    fn format(&self) -> PyAudioFormat {
        PyAudioFormat {
            inner: self.inner.format.clone(),
        }
    }
}

/// Python wrapper for WordTiming
#[pyclass(name = "WordTiming")]
#[derive(Clone)]
pub struct PyWordTiming {
    pub inner: CoreWordTiming,
}

#[pymethods]
impl PyWordTiming {
    #[new]
    #[pyo3(signature = (word, start_time, end_time, confidence=None))]
    fn new(word: String, start_time: f64, end_time: f64, confidence: Option<f32>) -> Self {
        Self {
            inner: CoreWordTiming {
                word,
                start: start_time as f32,
                end: end_time as f32,
                confidence: confidence.unwrap_or(1.0),
            },
        }
    }

    #[getter]
    fn word(&self) -> String {
        self.inner.word.clone()
    }

    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start as f64
    }

    #[getter]
    fn end_time(&self) -> f64 {
        self.inner.end as f64
    }

    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }
}

/// Python wrapper for TranscriptionResult
#[pyclass(name = "TranscriptionResult")]
#[derive(Clone)]
pub struct PyTranscriptionResult {
    pub inner: CoreTranscriptionResult,
}

#[pymethods]
impl PyTranscriptionResult {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    #[getter]
    fn language_detected(&self) -> String {
        self.inner.language_detected.clone().unwrap_or_else(|| "unknown".to_string())
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration.unwrap_or(0.0) as f64
    }

    #[getter]
    fn words(&self) -> Vec<PyWordTiming> {
        self.inner
            .words
            .as_ref()
            .map(|words| words.iter().map(|w| PyWordTiming { inner: w.clone() }).collect())
            .unwrap_or_else(Vec::new)
    }

    #[getter]
    fn is_final(&self) -> bool {
        true // Transcription results are always final
    }
}

/// Python wrapper for Voice
#[pyclass(name = "Voice")]
#[derive(Clone)]
pub struct PyVoice {
    pub inner: CoreVoice,
}

#[pymethods]
impl PyVoice {
    #[new]
    #[pyo3(signature = (voice_id, name, language, gender=None, description=None))]
    fn new(
        voice_id: String,
        name: String,
        language: String,
        gender: Option<String>,
        description: Option<String>,
    ) -> Self {
        Self {
            inner: CoreVoice {
                voice_id,
                name,
                language,
                description,
                gender,
                age: None,
                accent: None,
                style: None,
                use_case: None,
                preview_url: None,
                metadata: None,
            },
        }
    }

    #[getter]
    fn voice_id(&self) -> String {
        self.inner.voice_id.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn language(&self) -> String {
        self.inner.language.clone()
    }

    #[getter]
    fn gender(&self) -> Option<String> {
        self.inner.gender.clone()
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
}

/// Python wrapper for SynthesisResult
#[pyclass(name = "SynthesisResult")]
#[derive(Clone)]
pub struct PySynthesisResult {
    pub inner: CoreSynthesisResult,
}

#[pymethods]
impl PySynthesisResult {
    #[getter]
    fn audio_data(&self) -> Vec<u8> {
        self.inner.audio_data.clone()
    }

    #[getter]
    fn format(&self) -> String {
        self.inner.format.format.clone()
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.format.sample_rate
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration.unwrap_or(0.0) as f64
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes
    }

    #[getter]
    fn voice_used(&self) -> Option<String> {
        // Would need to be added to CoreSynthesisResult
        None
    }
}

/// Python wrapper for StreamingResult
#[pyclass(name = "StreamingResult")]
#[derive(Clone)]
pub struct PyStreamingResult {
    pub inner: CoreStreamingResult,
}

#[pymethods]
impl PyStreamingResult {
    #[new]
    #[pyo3(signature = (_session_id, is_final, text, confidence, _timestamp=None, _processing_time_ms=None))]
    fn new(
        _session_id: String,
        is_final: bool,
        text: String,
        confidence: f32,
        _timestamp: Option<String>,
        _processing_time_ms: Option<i32>,
    ) -> Self {
        Self {
            inner: CoreStreamingResult {
                session_id: uuid::Uuid::new_v4(),
                text,
                is_final,
                confidence,
                timestamp: chrono::Utc::now(),
                words: None,
                metadata: None,
            },
        }
    }

    #[getter]
    fn session_id(&self) -> String {
        self.inner.session_id.to_string()
    }

    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    #[getter]
    fn timestamp(&self) -> Option<String> {
        Some(self.inner.timestamp.to_rfc3339())
    }

    #[getter]
    fn processing_time_ms(&self) -> i32 {
        // Calculate processing time from metadata if available
        if let Some(metadata) = &self.inner.metadata {
            if let Some(time) = metadata.get("processing_time_ms") {
                if let Some(time_val) = time.as_i64() {
                    return time_val as i32;
                }
            }
        }
        0
    }
}

create_exception!(debabelizer, ProviderError, pyo3::exceptions::PyException);
create_exception!(debabelizer, AuthenticationError, ProviderError);
create_exception!(debabelizer, RateLimitError, ProviderError);
create_exception!(debabelizer, ConfigurationError, ProviderError);

/// Manager for active streaming sessions
struct StreamWrapper {
    stream: Arc<Mutex<Box<dyn SttStream>>>,
}

struct StreamManager {
    streams: Arc<Mutex<HashMap<uuid::Uuid, StreamWrapper>>>,
}

impl StreamManager {
    fn new() -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    async fn add_stream(&self, session_id: uuid::Uuid, stream: Box<dyn SttStream>) {
        let wrapper = StreamWrapper {
            stream: Arc::new(Mutex::new(stream)),
        };
        self.streams.lock().await.insert(session_id, wrapper);
    }
    
    async fn get_stream(&self, session_id: &uuid::Uuid) -> Option<Arc<Mutex<Box<dyn SttStream>>>> {
        self.streams.lock().await.get(session_id).map(|w| w.stream.clone())
    }
    
    async fn take_stream(&self, session_id: &uuid::Uuid) -> Option<Box<dyn SttStream>> {
        self.streams.lock().await.remove(session_id).and_then(|w| {
            // This is a hack to extract the stream from Arc<Mutex<Box<dyn SttStream>>>
            // In production, we should use get_stream instead
            match Arc::try_unwrap(w.stream) {
                Ok(mutex) => Some(mutex.into_inner()),
                Err(_) => None, // Arc is still in use elsewhere
            }
        })
    }
    
    async fn remove_stream(&self, session_id: &uuid::Uuid) -> Option<Box<dyn SttStream>> {
        self.take_stream(session_id).await
    }
}

/// Python async iterator for streaming results
#[pyclass]
struct PyStreamingResultIterator {
    session_id: uuid::Uuid,
    timeout: Option<std::time::Duration>,
    stream_manager: Arc<StreamManager>,
}

impl PyStreamingResultIterator {
    fn new(session_id: uuid::Uuid, timeout: Option<std::time::Duration>, stream_manager: Arc<StreamManager>) -> Self {
        Self {
            session_id,
            timeout,
            stream_manager,
        }
    }
}

#[pymethods]
impl PyStreamingResultIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(slf: PyRef<'_, Self>, py: Python<'py>) -> PyResult<Option<&'py PyAny>> {
        let stream_manager = slf.stream_manager.clone();
        let session_id = slf.session_id;
        let timeout = slf.timeout.unwrap_or(std::time::Duration::from_secs(1));
        
        // Use direct async approach without spawn_blocking
        let fut = pyo3_asyncio::tokio::future_into_py(py, async move {
            // Direct async implementation - no nested runtime
            
            // Get the stream reference 
            let stream_arc = match stream_manager.get_stream(&session_id).await {
                Some(s) => s,
                None => {
                    // Stream not found - end iterator
                    return Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>("Stream not found"));
                }
            };
            
            // Lock the stream and receive transcript with timeout
            let mut stream = stream_arc.lock().await;
            
            // Try to receive with timeout
            match tokio::time::timeout(timeout, stream.receive_transcript()).await {
                Ok(Ok(Some(streaming_result))) => {
                    // Got a real result
                    println!("📝 PYTHON WRAPPER: Received streaming result - text='{}', is_final={}, confidence={}", 
                        streaming_result.text, streaming_result.is_final, streaming_result.confidence);
                    let py_result = PyStreamingResult {
                        inner: streaming_result,
                    };
                    Ok(Python::with_gil(|py| py_result.into_py(py)))
                }
                Ok(Ok(None)) => {
                    // Stream ended normally
                    Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>("Stream ended"))
                }
                Ok(Err(e)) => {
                    // Stream error - end iterator
                    Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(format!("Stream error: {}", e)))
                }
                Err(_timeout) => {
                    // Timeout - return empty result to keep iterator alive
                    let keep_alive = PyStreamingResult {
                        inner: debabelizer_core::stt::StreamingResult {
                            session_id,
                            is_final: false,
                            text: String::new(),
                            confidence: 0.0,
                            timestamp: chrono::Utc::now(),
                            words: None,
                            metadata: Some(serde_json::json!({"type": "keep_alive", "source": "pyo3_timeout"})),
                        },
                    };
                    Ok(Python::with_gil(|py| keep_alive.into_py(py)))
                }
            }
        })?;
        
        Ok(Some(fut))
    }
}

/// Python wrapper for DebabelizerConfig
#[pyclass(name = "DebabelizerConfig")]
pub struct PyDebabelizerConfig {
    pub inner: CoreConfig,
}

#[pymethods]
impl PyDebabelizerConfig {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let config = if let Some(config_dict) = config {
            // Simple manual conversion for now - just parse the main structure
            let mut providers = std::collections::HashMap::new();
            
            for (key, value) in config_dict.iter() {
                let key_str = key.extract::<String>().unwrap_or_default();
                if key_str == "preferences" {
                    // Skip preferences for now
                    continue;
                }
                
                if let Ok(value_dict) = value.downcast::<PyDict>() {
                    let mut provider_config = std::collections::HashMap::new();
                    for (k, v) in value_dict.iter() {
                        let k_str = k.extract::<String>().unwrap_or_default();
                        let v_str = v.extract::<String>().unwrap_or_default();
                        provider_config.insert(k_str, serde_json::Value::String(v_str));
                    }
                    providers.insert(key_str, debabelizer::config::ProviderConfig::Simple(provider_config));
                }
            }
            
            debabelizer::config::DebabelizerConfig {
                preferences: debabelizer::config::Preferences::default(),
                providers,
            }
        } else {
            // Load config from environment by default, fall back to default if it fails
            CoreConfig::new().unwrap_or_else(|_| CoreConfig::default())
        };
        
        Ok(Self { inner: config })
    }
}

/// Python wrapper for VoiceProcessor
#[pyclass(name = "VoiceProcessor")]
pub struct PyVoiceProcessor {
    processor: Arc<Mutex<CoreProcessor>>,
    runtime: Runtime,
    stream_manager: Arc<StreamManager>,
}

#[pymethods]
impl PyVoiceProcessor {
    #[new]
    #[pyo3(signature = (stt_provider=None, tts_provider=None, config=None))]
    fn new(
        stt_provider: Option<String>,
        tts_provider: Option<String>,
        config: Option<&PyDebabelizerConfig>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        let cfg = config
            .map(|c| c.inner.clone())
            .unwrap_or_else(|| CoreConfig::new().unwrap_or_else(|_| CoreConfig::default()));

        let processor = runtime.block_on(async {
            let processor = CoreProcessor::with_config(cfg)?;
            
            // Set STT provider if specified
            if let Some(stt_name) = &stt_provider {
                processor.set_stt_provider(stt_name).await?;
            }
            
            // Set TTS provider if specified  
            if let Some(tts_name) = &tts_provider {
                processor.set_tts_provider(tts_name).await?;
            }
            
            Ok::<CoreProcessor, DebabelizerError>(processor)
        }).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create processor: {}", e))
        })?;

        Ok(Self { 
            processor: Arc::new(Mutex::new(processor)), 
            runtime,
            stream_manager: Arc::new(StreamManager::new()),
        })
    }

    /// Transcribe an audio file
    #[pyo3(signature = (file_path, _language=None, _language_hints=None))]
    fn transcribe_file(
        &mut self,
        file_path: String,
        _language: Option<String>,
        _language_hints: Option<Vec<String>>,
    ) -> PyResult<PyTranscriptionResult> {
        // For file transcription, read the file and call transcribe_audio
        let audio_data = std::fs::read(&file_path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e))
        })?;

        // Try to determine format from file extension
        let format = file_path
            .split('.')
            .last()
            .unwrap_or("wav")
            .to_lowercase();

        self.transcribe_audio(audio_data, Some(format), None, None, None)
    }

    /// Transcribe raw audio data
    #[pyo3(signature = (audio_data, audio_format=None, sample_rate=None, _language=None, _language_hints=None))]
    fn transcribe_audio(
        &mut self,
        audio_data: Vec<u8>,
        audio_format: Option<String>,
        sample_rate: Option<u32>,
        _language: Option<String>,
        _language_hints: Option<Vec<String>>,
    ) -> PyResult<PyTranscriptionResult> {
        let format = audio_format.unwrap_or_else(|| "wav".to_string());
        let rate = sample_rate.unwrap_or(16000);

        let audio_fmt = CoreAudioFormat {
            format,
            sample_rate: rate,
            channels: 1,
            bit_depth: Some(16),
        };

        let audio = AudioData {
            data: audio_data,
            format: audio_fmt,
        };

        let result = self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.transcribe(audio).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Transcription failed: {}", e))
        })?;

        Ok(PyTranscriptionResult { inner: result })
    }

    /// Transcribe audio data using AudioData object
    fn transcribe(&mut self, audio: PyAudioData) -> PyResult<PyTranscriptionResult> {
        let result = self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.transcribe(audio.inner).await
        }).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        Ok(PyTranscriptionResult { inner: result })
    }

    /// Transcribe audio chunk (compatibility method)
    #[pyo3(signature = (audio_data, audio_format=None, sample_rate=None, language=None))]
    fn transcribe_chunk(
        &mut self,
        audio_data: Vec<u8>,
        audio_format: Option<String>,
        sample_rate: Option<u32>,
        language: Option<String>,
    ) -> PyResult<PyTranscriptionResult> {
        // Just redirect to transcribe_audio
        self.transcribe_audio(audio_data, audio_format, sample_rate, language, None)
    }

    /// Synthesize speech from text
    #[pyo3(signature = (text, voice=None, language=None, audio_format=None, sample_rate=None))]
    fn synthesize(
        &mut self,
        text: String,
        voice: Option<PyRef<PyVoice>>,
        language: Option<String>,
        audio_format: Option<PyRef<PyAudioFormat>>,
        sample_rate: Option<u32>,
    ) -> PyResult<PySynthesisResult> {
        let voice_data = voice.map(|v| v.inner.clone()).unwrap_or_else(|| {
            let lang = language.clone().unwrap_or_else(|| "en".to_string());
            CoreVoice::new("default".to_string(), "Default Voice".to_string(), lang)
        });

        let format = audio_format.map(|f| f.inner.clone()).unwrap_or_else(|| {
            CoreAudioFormat::wav(sample_rate.unwrap_or(22050))
        });

        let options = CoreSynthesisOptions {
            voice: voice_data,
            model: None,
            speed: None,
            pitch: None,
            volume_gain_db: None,
            format,
            sample_rate,
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        };

        let result = self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.synthesize(&text, &options).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Synthesis failed: {}", e))
        })?;

        Ok(PySynthesisResult { inner: result })
    }

    /// Synthesize speech with text file output support (compatibility method)
    #[pyo3(signature = (text, output_file=None, voice_id=None, language=None, audio_format=None, sample_rate=None))]
    fn synthesize_text(
        &mut self,
        text: String,
        output_file: Option<String>,
        voice_id: Option<String>,
        language: Option<String>,
        audio_format: Option<String>,
        sample_rate: Option<u32>,
    ) -> PyResult<PySynthesisResult> {
        // Create a voice object if voice_id is provided
        let voice = voice_id.map(|id| {
            CoreVoice::new(id.clone(), id, language.unwrap_or_else(|| "en".to_string()))
        }).unwrap_or_else(|| {
            CoreVoice::new("default".to_string(), "Default Voice".to_string(), "en".to_string())
        });

        let format = CoreAudioFormat {
            format: audio_format.unwrap_or_else(|| "wav".to_string()),
            sample_rate: sample_rate.unwrap_or(22050),
            channels: 1,
            bit_depth: Some(16),
        };

        let options = CoreSynthesisOptions {
            voice,
            model: None,
            speed: None,
            pitch: None,
            volume_gain_db: None,
            format,
            sample_rate,
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        };

        let result = self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.synthesize(&text, &options).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Synthesis failed: {}", e))
        })?;

        // Write to file if requested
        if let Some(path) = output_file {
            std::fs::write(path, &result.audio_data).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to write audio file: {}", e))
            })?;
        }

        Ok(PySynthesisResult { inner: result })
    }

    /// Get available voices
    #[pyo3(signature = (language=None))]
    fn get_available_voices(&mut self, language: Option<String>) -> PyResult<Vec<PyVoice>> {
        let voices = self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.list_voices().await
        }).map_err(|e| {
            ProviderError::new_err(format!("Failed to get voices: {}", e))
        })?;

        let filtered_voices: Vec<PyVoice> = voices
            .into_iter()
            .filter(|v| language.as_ref().map_or(true, |lang| v.language.starts_with(lang)))
            .map(|v| PyVoice { inner: v })
            .collect();

        Ok(filtered_voices)
    }

    /// Start streaming transcription - creates real SttStream
    #[pyo3(signature = (audio_format="wav".to_string(), sample_rate=16000, language=None, _language_hints=None, enable_language_identification=None, has_pending_audio=None, interim_results=None))]
    #[allow(unused_variables)]
    fn start_streaming_transcription<'py>(
        &mut self,
        py: Python<'py>,
        audio_format: String,
        sample_rate: u32,
        language: Option<String>,
        _language_hints: Option<Vec<String>>,
        enable_language_identification: Option<bool>,
        has_pending_audio: Option<bool>,
        interim_results: Option<bool>,
    ) -> PyResult<&'py PyAny> {
        // Clone the Arcs to move into async block
        let processor = self.processor.clone();
        let stream_manager = self.stream_manager.clone();
        
        future_into_py(py, async move {
            // Create stream config 
            let session_id = uuid::Uuid::new_v4();
            println!("🚀 PYTHON WRAPPER: start_streaming_transcription called - session_id={}, format={}, sample_rate={}", 
                session_id, audio_format, sample_rate);
            let stream_config = debabelizer_core::StreamConfig {
                session_id,
                language: language.clone(),
                model: None,
                interim_results: interim_results.unwrap_or(true),
                format: debabelizer_core::AudioFormat {
                    format: audio_format.clone(),
                    sample_rate,
                    channels: 1,
                    bit_depth: Some(16),
                },
                punctuate: true,
                profanity_filter: false,
                diarization: false,
                metadata: None,
                enable_word_time_offsets: true,
                enable_automatic_punctuation: true,
                enable_language_identification: enable_language_identification.unwrap_or(false),
            };

            // Actually create the SttStream using the processor
            let processor = processor.lock().await;
            println!("🔒 PYTHON WRAPPER: Got processor lock, creating stream...");
            match processor.transcribe_stream(stream_config).await {
                Ok(stream) => {
                    println!("✅ PYTHON WRAPPER: Stream created successfully for session {}", session_id);
                    // Store the stream in the stream manager
                    stream_manager.add_stream(session_id, stream).await;
                    Ok(Python::with_gil(|py| session_id.to_string().to_object(py)))
                },
                Err(e) => Err(PyErr::new::<ProviderError, _>(format!("Failed to create stream: {}", e))),
            }
        })
    }

    /// Stream audio chunk - now async
    fn stream_audio<'py>(
        &mut self,
        py: Python<'py>,
        session_id: String,
        audio_chunk: Vec<u8>,
    ) -> PyResult<&'py PyAny> {
        // Validate session ID
        let session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        let stream_manager = self.stream_manager.clone();
        
        // Return async function that processes audio
        future_into_py(py, async move {
            println!("🎤 PYTHON WRAPPER: stream_audio called for session {} with {} bytes", session_id, audio_chunk.len());
            
            // Get the stream reference (doesn't remove it)
            let stream_arc = match stream_manager.get_stream(&session_uuid).await {
                Some(s) => s,
                None => {
                    return Err(PyErr::new::<ProviderError, _>(format!("No active stream for session {}", session_id)));
                }
            };
            
            // Lock the stream and send audio
            let mut stream = stream_arc.lock().await;
            match stream.send_audio(&audio_chunk).await {
                Ok(_) => {
                    println!("✅ PYTHON WRAPPER: Successfully sent {} bytes to stream", audio_chunk.len());
                    Ok(Python::with_gil(|py| py.None()))
                },
                Err(e) => {
                    println!("❌ PYTHON WRAPPER: Failed to send audio: {}", e);
                    Err(PyErr::new::<ProviderError, _>(format!("Failed to send audio: {}", e)))
                }
            }
        })
    }

    /// Stop streaming transcription - now async
    fn stop_streaming_transcription<'py>(
        &mut self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<&'py PyAny> {
        // Validate session ID
        let session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        let stream_manager = self.stream_manager.clone();
        
        // Return async function that stops streaming
        future_into_py(py, async move {
            // Remove and close the stream using take_stream (which still removes it)
            if let Some(mut stream) = stream_manager.take_stream(&session_uuid).await {
                // Close the stream properly
                match stream.close().await {
                    Ok(_) => Ok(Python::with_gil(|py| py.None())),
                    Err(e) => Err(PyErr::new::<ProviderError, _>(format!("Failed to close stream: {}", e)))
                }
            } else {
                // Stream already removed or doesn't exist - not an error
                Ok(Python::with_gil(|py| py.None()))
            }
        })
    }

    /// Get streaming results - returns actual stream from processor
    #[pyo3(signature = (session_id, timeout=None))]  
    fn get_streaming_results(
        &mut self,
        py: Python<'_>,
        session_id: String,
        timeout: Option<f64>,
    ) -> PyResult<PyObject> {
        // Validate session ID
        let session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        let stream_manager = self.stream_manager.clone();
        let timeout_duration = timeout.map(|t| std::time::Duration::from_secs_f64(t));
        
        // Create async iterator that reads from the actual SttStream
        let iter = PyStreamingResultIterator::new(session_uuid, timeout_duration, stream_manager);
        Ok(iter.into_py(py))
    }

    /// Start streaming TTS
    #[pyo3(signature = (_text, _voice_id=None, _audio_format=None))]
    fn start_streaming_tts(
        &mut self,
        _text: String,
        _voice_id: Option<String>,
        _audio_format: Option<String>,
    ) -> PyResult<String> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Streaming TTS not yet implemented in Rust version. Use synthesize() instead."
        ))
    }

    /// Get streaming audio
    fn get_streaming_audio(&mut self, _session_id: String) -> PyResult<Vec<u8>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Streaming TTS not yet implemented in Rust version. Use synthesize() instead."
        ))
    }

    /// End streaming synthesis
    fn end_streaming_synthesis(&mut self, _session_id: String) -> PyResult<bool> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Streaming TTS not yet implemented in Rust version. Use synthesize() instead."
        ))
    }

    /// Set STT provider
    fn set_stt_provider(&mut self, provider: String) -> PyResult<()> {
        self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.set_stt_provider(&provider).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Failed to set STT provider: {}", e))
        })?;

        Ok(())
    }

    /// Set TTS provider
    fn set_tts_provider(&mut self, provider: String) -> PyResult<()> {
        self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.set_tts_provider(&provider).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Failed to set TTS provider: {}", e))
        })?;

        Ok(())
    }

    /// Get usage statistics
    fn get_usage_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("stt_requests", 0)?;
            stats.set_item("tts_requests", 0)?;
            stats.set_item("stt_duration", 0.0)?;
            stats.set_item("tts_characters", 0)?;
            stats.set_item("cost_estimate", 0.0)?;
            stats.set_item("sessions_created", 0)?;
            Ok(stats.into())
        })
    }

    /// Reset usage statistics
    fn reset_usage_stats(&mut self) -> PyResult<()> {
        // Placeholder - usage stats would be handled in Rust processor
        Ok(())
    }

    /// Test providers
    fn test_providers(&mut self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results = PyDict::new(py);
            // Placeholder - would call actual test methods on Rust processor
            Ok(results.into())
        })
    }

    /// Get STT provider name
    #[getter]
    fn stt_provider_name(&self) -> Option<String> {
        self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.get_stt_provider_name().await
        })
    }

    /// Get TTS provider name
    #[getter]
    fn tts_provider_name(&self) -> Option<String> {
        self.runtime.block_on(async {
            let processor = self.processor.lock().await;
            processor.get_tts_provider_name().await
        })
    }

    /// Cleanup resources
    fn cleanup(&mut self) -> PyResult<()> {
        println!("🧹 PYTHON WRAPPER: Cleaning up VoiceProcessor resources");
        
        // Force cleanup all active streams
        self.runtime.block_on(async {
            let stream_map = self.stream_manager.streams.lock().await;
            let session_ids: Vec<uuid::Uuid> = stream_map.keys().cloned().collect();
            drop(stream_map);
            
            for session_id in session_ids {
                if let Some(mut stream) = self.stream_manager.take_stream(&session_id).await {
                    println!("🛑 PYTHON WRAPPER: Force closing stream {}", session_id);
                    let _ = stream.close().await;
                }
            }
        });
        
        println!("✅ PYTHON WRAPPER: VoiceProcessor cleanup complete");
        Ok(())
    }
}

impl Drop for PyVoiceProcessor {
    fn drop(&mut self) {
        println!("🧹 PYTHON WRAPPER: PyVoiceProcessor dropping - ensuring all streams are cleaned up");
        
        // Force cleanup all active streams on drop to prevent orphaned processes
        self.runtime.block_on(async {
            let stream_map = self.stream_manager.streams.lock().await;
            let session_ids: Vec<uuid::Uuid> = stream_map.keys().cloned().collect();
            drop(stream_map);
            
            for session_id in session_ids {
                if let Some(mut stream) = self.stream_manager.take_stream(&session_id).await {
                    println!("🚨 PYTHON WRAPPER: Emergency cleanup - closing stream {}", session_id);
                    let _ = stream.close().await;
                }
            }
        });
        
        println!("✅ PYTHON WRAPPER: PyVoiceProcessor drop cleanup complete");
    }
}

/// Convenience function to create a VoiceProcessor
#[pyfunction]
#[pyo3(signature = (stt_provider="soniox", tts_provider="elevanlabs", **config))]
fn create_processor(
    stt_provider: &str,
    tts_provider: &str,
    config: Option<&PyDict>,
) -> PyResult<PyVoiceProcessor> {
    let py_config = if let Some(_cfg) = config {
        Some(PyDebabelizerConfig::new(Some(_cfg))?)
    } else {
        None
    };

    PyVoiceProcessor::new(
        Some(stt_provider.to_string()), 
        Some(tts_provider.to_string()), 
        py_config.as_ref()
    )
}

/// Python module definition
#[pymodule]
fn _internal(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAudioFormat>()?;
    m.add_class::<PyAudioData>()?;
    m.add_class::<PyWordTiming>()?;
    m.add_class::<PyTranscriptionResult>()?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PySynthesisResult>()?;
    m.add_class::<PyStreamingResult>()?;
    m.add_class::<PyDebabelizerConfig>()?;
    m.add_class::<PyVoiceProcessor>()?;
    
    m.add("ProviderError", py.get_type::<ProviderError>())?;
    m.add("AuthenticationError", py.get_type::<AuthenticationError>())?;
    m.add("RateLimitError", py.get_type::<RateLimitError>())?;
    m.add("ConfigurationError", py.get_type::<ConfigurationError>())?;
    
    m.add_function(wrap_pyfunction!(create_processor, m)?)?;
    
    Ok(())
}