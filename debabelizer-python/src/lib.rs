use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::create_exception;
use tokio::runtime::Runtime;

// Import the Rust debabelizer types
use debabelizer_core::{
    AudioData, AudioFormat as CoreAudioFormat, TranscriptionResult as CoreTranscriptionResult,
    SynthesisResult as CoreSynthesisResult, Voice as CoreVoice, WordTiming as CoreWordTiming,
    StreamingResult as CoreStreamingResult, SynthesisOptions as CoreSynthesisOptions, 
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
        None // Placeholder - would need to convert from Option<DateTime>
    }

    #[getter]
    fn processing_time_ms(&self) -> i32 {
        0 // Placeholder
    }
}

create_exception!(debabelizer, ProviderError, pyo3::exceptions::PyException);
create_exception!(debabelizer, AuthenticationError, ProviderError);
create_exception!(debabelizer, RateLimitError, ProviderError);
create_exception!(debabelizer, ConfigurationError, ProviderError);

/// Python wrapper for DebabelizerConfig
#[pyclass(name = "DebabelizerConfig")]
pub struct PyDebabelizerConfig {
    pub inner: CoreConfig,
}

#[pymethods]
impl PyDebabelizerConfig {
    #[new]
    #[pyo3(signature = (_config=None))]
    fn new(_config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // For now, just use default config - would need to parse the dict
        let config = CoreConfig::default();
        Ok(Self { inner: config })
    }
}

/// Python wrapper for VoiceProcessor
#[pyclass(name = "VoiceProcessor")]
pub struct PyVoiceProcessor {
    processor: CoreProcessor,
    runtime: Runtime,
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
            .unwrap_or_else(|| CoreConfig::default());

        let processor = runtime.block_on(async {
            let mut processor = CoreProcessor::with_config(cfg)?;
            
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

        Ok(Self { processor, runtime })
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
            self.processor.transcribe(audio).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Transcription failed: {}", e))
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
    #[pyo3(signature = (text, voice=None, audio_format=None, sample_rate=None))]
    fn synthesize(
        &mut self,
        text: String,
        voice: Option<PyRef<PyVoice>>,
        audio_format: Option<PyRef<PyAudioFormat>>,
        sample_rate: Option<u32>,
    ) -> PyResult<PySynthesisResult> {
        let voice_data = voice.map(|v| v.inner.clone()).unwrap_or_else(|| {
            CoreVoice::new("default".to_string(), "Default Voice".to_string(), "en".to_string())
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
            self.processor.synthesize(&text, &options).await
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
            self.processor.synthesize(&text, &options).await
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
            self.processor.list_voices().await
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

    /// Start streaming transcription
    #[pyo3(signature = (audio_format="wav".to_string(), sample_rate=16000, language=None, _language_hints=None, enable_language_identification=None, has_pending_audio=None))]
    #[allow(unused_variables)]
    fn start_streaming_transcription(
        &mut self,
        audio_format: String,
        sample_rate: u32,
        language: Option<String>,
        _language_hints: Option<Vec<String>>,
        enable_language_identification: Option<bool>,
        has_pending_audio: Option<bool>,
    ) -> PyResult<String> {
        self.runtime.block_on(async {
            // Create stream config
            let session_id = uuid::Uuid::new_v4();
            let stream_config = debabelizer_core::StreamConfig {
                session_id,
                language: language.clone(),
                model: None,
                format: debabelizer_core::AudioFormat {
                    format: audio_format.clone(),
                    sample_rate,
                    channels: 1,
                    bit_depth: Some(16),
                },
                interim_results: true,
                punctuate: true,
                profanity_filter: false,
                diarization: false,
                metadata: None,
                enable_word_time_offsets: true,
                enable_automatic_punctuation: true,
                enable_language_identification: enable_language_identification.unwrap_or(false),
            };

            // Create STT stream with the processor  
            match self.processor.create_stt_stream(stream_config).await {
                Ok(_stream) => {
                    // The stream is created successfully, but we need to manage it.
                    // For now, we'll let the session manager handle the stream lifecycle
                    // and just return the session ID
                    Ok(session_id.to_string())
                },
                Err(e) => Err(PyErr::new::<ProviderError, _>(format!("Failed to start streaming: {}", e))),
            }
        })
    }

    /// Stream audio chunk
    fn stream_audio(&mut self, session_id: String, _audio_chunk: Vec<u8>) -> PyResult<()> {
        // For API compatibility, accept the parameters but implement streaming differently
        // The actual streaming would be handled by the SttStream trait object
        // For now, just validate the session ID and return success
        let _session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        // TODO: In a full implementation, we would route this to the appropriate SttStream
        // For now, just return success to maintain API compatibility
        Ok(())
    }

    /// Stop streaming transcription
    fn stop_streaming_transcription(&mut self, session_id: String) -> PyResult<()> {
        // For API compatibility, accept the session ID but implement cleanup differently
        let _session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        // TODO: In a full implementation, we would close the SttStream for this session
        // For now, just return success to maintain API compatibility
        Ok(())
    }

    /// Get streaming results
    #[pyo3(signature = (session_id, timeout=None))]
    fn get_streaming_results(
        &mut self,
        session_id: String,
        timeout: Option<f64>,
    ) -> PyResult<Option<PyStreamingResult>> {
        // For API compatibility, accept the parameters but implement differently
        let _session_uuid = uuid::Uuid::parse_str(&session_id)
            .map_err(|e| PyErr::new::<ProviderError, _>(format!("Invalid session ID: {}", e)))?;
        
        let _timeout_duration = timeout.map(|t| std::time::Duration::from_secs_f64(t));
        
        // TODO: In a full implementation, we would read from the SttStream for this session
        // For now, just return None to indicate no results available
        Ok(None)
    }

    /// Start streaming TTS
    #[pyo3(signature = (_text, _voice_id=None, _audio_format=None))]
    fn start_streaming_tts(
        &mut self,
        _text: String,
        _voice_id: Option<String>,
        _audio_format: Option<String>,
    ) -> PyResult<String> {
        // Placeholder - would return session ID
        Ok("tts-session-123".to_string())
    }

    /// Get streaming audio
    fn get_streaming_audio(&mut self, _session_id: String) -> PyResult<Vec<u8>> {
        // Placeholder - would return audio chunk
        Ok(vec![])
    }

    /// End streaming synthesis
    fn end_streaming_synthesis(&mut self, _session_id: String) -> PyResult<bool> {
        // Placeholder
        Ok(true)
    }

    /// Set STT provider
    fn set_stt_provider(&mut self, provider: String) -> PyResult<()> {
        self.runtime.block_on(async {
            self.processor.set_stt_provider(&provider).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Failed to set STT provider: {}", e))
        })?;

        Ok(())
    }

    /// Set TTS provider
    fn set_tts_provider(&mut self, provider: String) -> PyResult<()> {
        self.runtime.block_on(async {
            self.processor.set_tts_provider(&provider).await
        }).map_err(|e| {
            ProviderError::new_err(format!("Failed to set TTS provider: {}", e))
        })?;

        Ok(())
    }

    /// Get usage statistics
    fn get_usage_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = PyDict::new_bound(py);
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
            let results = PyDict::new_bound(py);
            // Placeholder - would call actual test methods on Rust processor
            Ok(results.into())
        })
    }

    /// Get STT provider name
    #[getter]
    fn stt_provider_name(&self) -> Option<String> {
        // Would need to be implemented in Rust processor
        None
    }

    /// Get TTS provider name
    #[getter]
    fn tts_provider_name(&self) -> Option<String> {
        // Would need to be implemented in Rust processor
        None
    }

    /// Cleanup resources
    fn cleanup(&mut self) -> PyResult<()> {
        // Cleanup would be handled automatically when the processor is dropped
        Ok(())
    }
}

/// Convenience function to create a VoiceProcessor
#[pyfunction]
#[pyo3(signature = (stt_provider="soniox", tts_provider="elevanlabs", **config))]
fn create_processor(
    stt_provider: &str,
    tts_provider: &str,
    config: Option<&Bound<'_, PyDict>>,
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
fn _internal(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAudioFormat>()?;
    m.add_class::<PyWordTiming>()?;
    m.add_class::<PyTranscriptionResult>()?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PySynthesisResult>()?;
    m.add_class::<PyStreamingResult>()?;
    m.add_class::<PyDebabelizerConfig>()?;
    m.add_class::<PyVoiceProcessor>()?;
    
    m.add("ProviderError", py.get_type_bound::<ProviderError>())?;
    m.add("AuthenticationError", py.get_type_bound::<AuthenticationError>())?;
    m.add("RateLimitError", py.get_type_bound::<RateLimitError>())?;
    m.add("ConfigurationError", py.get_type_bound::<ConfigurationError>())?;
    
    m.add_function(wrap_pyfunction!(create_processor, m)?)?;
    
    Ok(())
}