use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::{create_exception, PyResult, Bound};
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use debabelizer::{
    AudioData, AudioFormat, DebabelizerConfig, DebabelizerError, SynthesisOptions, 
    TranscriptionResult, SynthesisResult, VoiceProcessor, Voice, WordTiming
};

/// Python wrapper for AudioFormat
#[pyclass]
#[derive(Clone)]
pub struct PyAudioFormat {
    inner: AudioFormat,
}

#[pymethods]
impl PyAudioFormat {
    #[new]
    fn new(format: String, sample_rate: u32, channels: u8, bit_depth: Option<u16>) -> Self {
        Self {
            inner: AudioFormat {
                format,
                sample_rate,
                channels,
                bit_depth,
            }
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
    fn bit_depth(&self) -> Option<u16> {
        self.inner.bit_depth
    }
}

/// Python wrapper for AudioData
#[pyclass]
pub struct PyAudioData {
    inner: AudioData,
}

#[pymethods]
impl PyAudioData {
    #[new]
    fn new(data: &Bound<'_, PyBytes>, format: PyAudioFormat) -> Self {
        Self {
            inner: AudioData {
                data: data.as_bytes().to_vec(),
                format: format.inner,
            }
        }
    }

    #[getter]
    fn data(&self, py: Python) -> PyObject {
        PyBytes::new_bound(py, &self.inner.data).into()
    }

    #[getter]
    fn format(&self) -> PyAudioFormat {
        PyAudioFormat { inner: self.inner.format.clone() }
    }
}

/// Python wrapper for WordTiming
#[pyclass]
#[derive(Clone)]
pub struct PyWordTiming {
    inner: WordTiming,
}

#[pymethods]
impl PyWordTiming {
    #[getter]
    fn word(&self) -> String {
        self.inner.word.clone()
    }

    #[getter]
    fn start_time(&self) -> f32 {
        self.inner.start
    }

    #[getter]
    fn end_time(&self) -> f32 {
        self.inner.end
    }

    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }
}

/// Python wrapper for TranscriptionResult
#[pyclass]
pub struct PyTranscriptionResult {
    inner: TranscriptionResult,
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
    fn language_detected(&self) -> Option<String> {
        self.inner.language_detected.clone()
    }

    #[getter]
    fn duration(&self) -> Option<f32> {
        self.inner.duration
    }

    #[getter]
    fn words(&self) -> Option<Vec<PyWordTiming>> {
        self.inner.words.as_ref().map(|words| {
            words.iter().map(|w| PyWordTiming { inner: w.clone() }).collect()
        })
    }
}

/// Python wrapper for Voice
#[pyclass]
#[derive(Clone)]
pub struct PyVoice {
    inner: Voice,
}

#[pymethods]
impl PyVoice {
    #[new]
    fn new(voice_id: String, name: String, language: String) -> Self {
        Self {
            inner: Voice::new(voice_id, name, language)
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
#[pyclass]
pub struct PySynthesisResult {
    inner: SynthesisResult,
}

#[pymethods]
impl PySynthesisResult {
    #[getter]
    fn audio_data(&self, py: Python) -> PyObject {
        PyBytes::new_bound(py, &self.inner.audio_data).into()
    }

    #[getter]
    fn format(&self) -> PyAudioFormat {
        PyAudioFormat { inner: self.inner.format.clone() }
    }

    #[getter]
    fn duration(&self) -> Option<f32> {
        self.inner.duration
    }

    #[getter]
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes
    }
}

/// Python wrapper for SynthesisOptions
#[pyclass]
#[derive(Clone)]
pub struct PySynthesisOptions {
    inner: SynthesisOptions,
}

#[pymethods]
impl PySynthesisOptions {
    #[new]
    #[pyo3(signature = (voice=None, model=None, speed=None, pitch=None, volume_gain_db=None, format=None))]
    fn new(
        voice: Option<PyVoice>,
        model: Option<String>,
        speed: Option<f32>,
        pitch: Option<f32>,
        volume_gain_db: Option<f32>,
        format: Option<PyAudioFormat>,
    ) -> Self {
        let default_voice = Voice::new("alloy".to_string(), "Alloy".to_string(), "en-US".to_string());
        let voice = voice.map(|v| v.inner).unwrap_or(default_voice);
        let format = format.map(|f| f.inner).unwrap_or_default();
        
        Self {
            inner: SynthesisOptions {
                voice,
                model,
                speed,
                pitch,
                volume_gain_db,
                format,
                sample_rate: None,
                metadata: None,
                voice_id: None,
                stability: None,
                similarity_boost: None,
                output_format: None,
            }
        }
    }

    #[getter]
    fn voice(&self) -> PyVoice {
        PyVoice { inner: self.inner.voice.clone() }
    }

    #[setter]
    fn set_voice(&mut self, voice: PyVoice) {
        self.inner.voice = voice.inner;
    }

    #[getter]
    fn speed(&self) -> Option<f32> {
        self.inner.speed
    }

    #[setter]
    fn set_speed(&mut self, speed: Option<f32>) {
        self.inner.speed = speed;
    }

    #[getter]
    fn pitch(&self) -> Option<f32> {
        self.inner.pitch
    }

    #[setter]
    fn set_pitch(&mut self, pitch: Option<f32>) {
        self.inner.pitch = pitch;
    }

    #[getter]
    fn volume(&self) -> Option<f32> {
        self.inner.volume_gain_db
    }

    #[setter]
    fn set_volume(&mut self, volume: Option<f32>) {
        self.inner.volume_gain_db = volume;
    }

    #[getter]
    fn format(&self) -> PyAudioFormat {
        PyAudioFormat { inner: self.inner.format.clone() }
    }

    #[setter]
    fn set_format(&mut self, format: PyAudioFormat) {
        self.inner.format = format.inner;
    }
}

/// Python wrapper for VoiceProcessor
#[pyclass]
pub struct PyVoiceProcessor {
    inner: Arc<VoiceProcessor>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyVoiceProcessor {
    #[new]
    #[pyo3(signature = (config=None, stt_provider=None, tts_provider=None))]
    fn new(
        config: Option<&Bound<'_, PyDict>>,
        stt_provider: Option<String>,
        tts_provider: Option<String>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e)))?;

        let debabelizer_config = if let Some(config_dict) = config {
            let config_map: HashMap<String, Value> = config_dict
                .iter()
                .map(|(k, v)| {
                    let key = k.to_string();
                    let value = python_to_serde_value(&v).unwrap_or(Value::Null);
                    (key, value)
                })
                .collect();
            DebabelizerConfig::from_map(config_map)
                .map_err(|e| PyValueError::new_err(format!("Invalid configuration: {}", e)))?
        } else {
            DebabelizerConfig::from_env()
                .map_err(|e| PyValueError::new_err(format!("Failed to load configuration: {}", e)))?
        };

        let processor = runtime.block_on(async {
            let mut builder = VoiceProcessor::builder().config(debabelizer_config);
            
            if let Some(stt) = stt_provider {
                builder = builder.stt_provider(&stt);
            }
            
            if let Some(tts) = tts_provider {
                builder = builder.tts_provider(&tts);
            }
            
            builder.build().await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to create processor: {}", e)))?;

        Ok(Self {
            inner: Arc::new(processor),
            runtime,
        })
    }

    /// Transcribe audio to text
    fn transcribe(&self, audio: &PyAudioData) -> PyResult<PyTranscriptionResult> {
        let audio_data = audio.inner.clone();
        let processor = self.inner.clone();
        
        let result = self.runtime.block_on(async {
            processor.transcribe(audio_data).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Transcription failed: {}", e)))?;

        Ok(PyTranscriptionResult { inner: result })
    }

    /// Synthesize text to speech
    #[pyo3(signature = (text, options=None))]
    fn synthesize(&self, text: String, options: Option<PySynthesisOptions>) -> PyResult<PySynthesisResult> {
        let synthesis_options = options.map(|o| o.inner).unwrap_or_else(|| {
            let default_voice = Voice::new("alloy".to_string(), "Alloy".to_string(), "en-US".to_string());
            SynthesisOptions::new(default_voice)
        });
        let processor = self.inner.clone();
        
        let result = self.runtime.block_on(async {
            processor.synthesize(&text, &synthesis_options).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Synthesis failed: {}", e)))?;

        Ok(PySynthesisResult { inner: result })
    }

    /// Get available voices for TTS
    fn get_available_voices(&self) -> PyResult<Vec<PyVoice>> {
        let _processor = self.inner.clone();
        
        let voices = self.runtime.block_on(async {
            // For now, return a basic set of voices since the actual implementation might not have this method
            Ok(vec![
                Voice::new("alloy".to_string(), "Alloy".to_string(), "en-US".to_string()),
                Voice::new("echo".to_string(), "Echo".to_string(), "en-US".to_string()),
                Voice::new("fable".to_string(), "Fable".to_string(), "en-US".to_string()),
            ])
        }).map_err(|e: DebabelizerError| PyRuntimeError::new_err(format!("Failed to get voices: {}", e)))?;

        Ok(voices.into_iter().map(|v| PyVoice { inner: v }).collect())
    }

    /// Check if processor has been configured
    fn is_configured(&self) -> bool {
        true // Simplified for now
    }
}

/// Convert Python object to serde_json::Value
fn python_to_serde_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(Value::Number(serde_json::Number::from_f64(f).unwrap_or_else(|| 0.into())))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = k.to_string();
            let value = python_to_serde_value(&v)?;
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    } else {
        Ok(Value::Null)
    }
}

create_exception!(debabelizer_python, DebabelizerException, PyRuntimeError);

/// Python module definition
#[pymodule]
fn _debabelizer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAudioFormat>()?;
    m.add_class::<PyAudioData>()?;
    m.add_class::<PyWordTiming>()?;
    m.add_class::<PyTranscriptionResult>()?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PySynthesisResult>()?;
    m.add_class::<PySynthesisOptions>()?;
    m.add_class::<PyVoiceProcessor>()?;
    m.add("DebabelizerException", _py.get_type_bound::<DebabelizerException>())?;
    Ok(())
}