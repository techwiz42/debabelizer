use async_trait::async_trait;
pub use debabelizer_core::*;

#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Simple(std::collections::HashMap<String, serde_json::Value>),
}

impl ProviderConfig {
    pub fn get_api_key(&self) -> Option<String> {
        match self {
            Self::Simple(map) => map
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }
    }
    
    pub fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        match self {
            Self::Simple(map) => map.get(key),
        }
    }
}
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use uuid::Uuid;

const SONIOX_WS_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";

#[derive(Debug)]
enum WebSocketCommand {
    SendAudio(Vec<u8>),
    EndAudio,  // Signal no more audio is coming
    Close,
    Shutdown,  // Force shutdown the background task
}

#[derive(Debug)]
enum WebSocketMessage {
    Transcript(StreamingResult),
    Error(String),
    Closed,
}

#[derive(Debug)]
pub struct SonioxProvider {
    api_key: String,
    model: String,
    auto_detect_language: bool,
}

impl SonioxProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Soniox API key not found".to_string()))?;
        
        let model = config
            .get_value("model")
            .and_then(|v| v.as_str())
            .unwrap_or("stt-rt-preview-v2")
            .to_string();
        
        let auto_detect_language = config
            .get_value("auto_detect_language")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        
        Ok(Self { 
            api_key, 
            model,
            auto_detect_language,
        })
    }
}

#[async_trait]
impl SttProvider for SonioxProvider {
    fn name(&self) -> &str {
        "soniox"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        // For batch transcription, we'll use streaming and collect results
        let config = StreamConfig {
            session_id: Uuid::new_v4(),
            format: audio.format.clone(),
            interim_results: false,
            language: if self.auto_detect_language { None } else { Some(self.model.clone()) },
            ..Default::default()
        };
        
        let mut stream = self.transcribe_stream(config).await?;
        
        // Send all audio at once
        stream.send_audio(&audio.data).await?;
        stream.close().await?;
        
        // Collect all results
        let mut full_text = String::new();
        let mut words = Vec::new();
        let mut confidence_sum = 0.0;
        let mut confidence_count = 0;
        let mut detected_language = None;
        
        while let Some(result) = stream.receive_transcript().await? {
            if result.is_final {
                full_text.push_str(&result.text);
                full_text.push(' ');
                
                if let Some(result_words) = result.words {
                    words.extend(result_words);
                }
                
                confidence_sum += result.confidence;
                confidence_count += 1;
                
                // Extract language from metadata if available
                if detected_language.is_none() {
                    if let Some(metadata) = &result.metadata {
                        if let Some(lang) = metadata.get("language").and_then(|v| v.as_str()) {
                            detected_language = Some(lang.to_string());
                        }
                    }
                }
            }
        }
        
        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            confidence: if confidence_count > 0 {
                confidence_sum / confidence_count as f32
            } else {
                0.0
            },
            language_detected: detected_language.or_else(|| {
                if !self.auto_detect_language {
                    Some(self.model.clone())
                } else {
                    None
                }
            }),
            duration: None,
            words: if words.is_empty() { None } else { Some(words) },
            metadata: None,
        })
    }
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let stream = SonioxStream::new(
            self.api_key.clone(), 
            self.model.clone(), 
            self.auto_detect_language,
            config
        ).await?;
        Ok(Box::new(stream))
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model {
                id: "auto".to_string(),
                name: "Auto-detect Language".to_string(),
                languages: vec!["multi".to_string()],
                capabilities: vec![
                    "streaming".to_string(), 
                    "timestamps".to_string(),
                    "language-detection".to_string()
                ],
            },
            Model {
                id: "en".to_string(),
                name: "English".to_string(),
                languages: vec!["en".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "es".to_string(),
                name: "Spanish".to_string(),
                languages: vec!["es".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "fr".to_string(),
                name: "French".to_string(),
                languages: vec!["fr".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "de".to_string(),
                name: "German".to_string(),
                languages: vec!["de".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "it".to_string(),
                name: "Italian".to_string(),
                languages: vec!["it".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "pt".to_string(),
                name: "Portuguese".to_string(),
                languages: vec!["pt".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "nl".to_string(),
                name: "Dutch".to_string(),
                languages: vec!["nl".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ru".to_string(),
                name: "Russian".to_string(),
                languages: vec!["ru".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "zh".to_string(),
                name: "Chinese".to_string(),
                languages: vec!["zh".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ja".to_string(),
                name: "Japanese".to_string(),
                languages: vec!["ja".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ko".to_string(),
                name: "Korean".to_string(),
                languages: vec!["ko".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "hi".to_string(),
                name: "Hindi".to_string(),
                languages: vec!["hi".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
        ])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(16000),
            AudioFormat::wav(8000),
            AudioFormat::wav(48000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

struct SonioxStream {
    session_id: Uuid,
    command_tx: mpsc::UnboundedSender<WebSocketCommand>,
    result_rx: Arc<Mutex<mpsc::UnboundedReceiver<WebSocketMessage>>>,
    _task_handle: tokio::task::JoinHandle<()>,  // Store task handle for proper cleanup
}

impl SonioxStream {
    async fn new(
        api_key: String, 
        model: String, 
        auto_detect_language: bool,
        config: StreamConfig
    ) -> Result<Self> {
        // Create channels for communication with background WebSocket handler
        let (command_tx, command_rx) = mpsc::unbounded_channel::<WebSocketCommand>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<WebSocketMessage>();
        
        // Connect to WebSocket with API key in URL
        let url = format!("{}?api_key={}", SONIOX_WS_URL, api_key);
        let (ws_stream, _) = connect_async(&url)
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        // Create initial configuration message
        let init_message = json!({
            "api_key": api_key,
            "audio_format": "pcm_s16le",
            "sample_rate": config.format.sample_rate,
            "num_channels": config.format.channels,
            "model": model,
            "enable_language_identification": auto_detect_language || config.language.is_none(),
            "include_nonfinal": config.interim_results,
        });
        
        let session_id = config.session_id;
        
        // Start background WebSocket handler task
        let handler_task = tokio::spawn(websocket_handler(
            ws_stream,
            init_message,
            session_id,
            command_rx,
            result_tx,
        ));
        
        Ok(Self {
            session_id,
            command_tx,
            result_rx: Arc::new(Mutex::new(result_rx)),
            _task_handle: handler_task,
        })
    }
}

#[async_trait]
impl SttStream for SonioxStream {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()> {
        println!("🎵 SONIOX: send_audio called with {} bytes", chunk.len());
        
        // If chunk is empty, send EndAudio command instead
        let command = if chunk.is_empty() {
            println!("🏁 SONIOX: Empty chunk detected, sending EndAudio signal");
            WebSocketCommand::EndAudio
        } else {
            WebSocketCommand::SendAudio(chunk.to_vec())
        };
        
        self.command_tx
            .send(command)
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        println!("✅ SONIOX: Audio command sent to WebSocket handler");
        Ok(())
    }
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>> {
        println!("🔍 SONIOX: receive_transcript called, waiting for result...");
        let message = {
            let mut rx = self.result_rx.lock().await;
            rx.recv().await
        };
        
        match message {
            Some(WebSocketMessage::Transcript(result)) => {
                println!("📝 SONIOX: Received transcript result - text='{}', is_final={}", result.text, result.is_final);
                Ok(Some(result))
            },
            Some(WebSocketMessage::Error(error)) => {
                println!("❌ SONIOX: Received error: {}", error);
                Err(DebabelizerError::Provider(ProviderError::Network(error)))
            }
            Some(WebSocketMessage::Closed) => {
                println!("🛑 SONIOX: Stream closed");
                Ok(None)
            },
            None => {
                println!("📪 SONIOX: Channel closed");
                Ok(None) // Channel closed
            }
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        println!("🛑 SONIOX: Closing stream for session {}", self.session_id);
        
        // Send shutdown command to background task
        let _ = self.command_tx.send(WebSocketCommand::Shutdown);
        
        // The task handle will be dropped automatically when SonioxStream is dropped,
        // which will force the background task to terminate
        println!("✅ SONIOX: Stream close complete for session {}", self.session_id);
        Ok(())
    }
    
    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

impl Drop for SonioxStream {
    fn drop(&mut self) {
        println!("🧹 RUST: SonioxStream dropping - aborting background task for session {}", self.session_id);
        
        // Send shutdown command if channel is still open
        let _ = self.command_tx.send(WebSocketCommand::Shutdown);
        
        // Abort the background task to prevent orphaned processes
        self._task_handle.abort();
        
        println!("✅ RUST: Background task aborted for session {}", self.session_id);
    }
}

// Standalone WebSocket handler function (runs as background task)
async fn websocket_handler(
    mut ws_stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    init_message: serde_json::Value,
    session_id: Uuid,
    mut command_rx: mpsc::UnboundedReceiver<WebSocketCommand>,
    result_tx: mpsc::UnboundedSender<WebSocketMessage>
) {
    println!("🔧 RUST: Starting Soniox WebSocket background handler for session {}", session_id);
    
    // Send initial configuration
    println!("📤 RUST: Sending initial configuration to Soniox...");
    if let Err(e) = ws_stream.send(Message::Text(init_message.to_string())).await {
        let _ = result_tx.send(WebSocketMessage::Error(format!("Failed to send config: {}", e)));
        return;
    }
    println!("✅ RUST: Configuration sent successfully");
    
    // Wait for handshake response
    match ws_stream.next().await {
        Some(Ok(Message::Text(text))) => {
            println!("📥 RUST: Received Soniox handshake: {}", text);
            // Check for immediate errors in handshake
            if text.contains("\"error_code\"") || text.contains("Missing API key") {
                let _ = result_tx.send(WebSocketMessage::Error("Authentication failed".to_string()));
                return;
            }
            println!("✅ RUST: Soniox handshake successful - entering main loop");
        }
        Some(Ok(msg)) => {
            println!("🔍 RUST: Unexpected handshake message: {:?}", msg);
        }
        Some(Err(e)) => {
            let _ = result_tx.send(WebSocketMessage::Error(format!("Handshake error: {}", e)));
            return;
        }
        None => {
            let _ = result_tx.send(WebSocketMessage::Error("Connection closed during handshake".to_string()));
            return;
        }
    }
    
    // Main event loop using tokio::select!
    let mut messages_received = 0;
    loop {
        println!("🔄 RUST: Main loop iteration starting - task processing commands/messages");
        
        tokio::select! {
            // Add a timeout branch to see if we're stuck waiting
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(10)) => {
                println!("⏰ RUST: No activity for 10 seconds, messages received so far: {}", messages_received);
                // Send a keep-alive result to prevent iterator timeout
                let result = StreamingResult {
                    session_id,
                    is_final: false,
                    text: String::new(),
                    confidence: 0.0,
                    timestamp: chrono::Utc::now(),
                    words: None,
                    metadata: Some(json!({"type": "timeout_keepalive"})),
                };
                let _ = result_tx.send(WebSocketMessage::Transcript(result));
            }
            // Handle commands from the stream interface
            command = command_rx.recv() => {
                match command {
                    Some(WebSocketCommand::SendAudio(data)) => {
                        println!("📤 RUST: Sending {} bytes of audio to Soniox via WebSocket", data.len());
                        if let Err(e) = ws_stream.send(Message::Binary(data)).await {
                            let _ = result_tx.send(WebSocketMessage::Error(format!("Failed to send audio: {}", e)));
                            break;
                        }
                    }
                    Some(WebSocketCommand::EndAudio) => {
                        println!("🏁 RUST: Signaling end of audio to Soniox");
                        // Send empty message to signal end of audio (like Python implementation)
                        if let Err(e) = ws_stream.send(Message::Binary(vec![])).await {
                            let _ = result_tx.send(WebSocketMessage::Error(format!("Failed to send end-of-audio: {}", e)));
                            break;
                        }
                        println!("✅ RUST: End-of-audio signal sent successfully");
                    }
                    Some(WebSocketCommand::Close) => {
                        println!("🛑 RUST: Closing WebSocket connection");
                        let _ = ws_stream.close(None).await;
                        let _ = result_tx.send(WebSocketMessage::Closed);
                        break;
                    }
                    Some(WebSocketCommand::Shutdown) => {
                        println!("🚨 RUST: Shutdown command received - force terminating WebSocket handler");
                        let _ = ws_stream.close(None).await;
                        let _ = result_tx.send(WebSocketMessage::Closed);
                        break;
                    }
                    None => {
                        println!("📪 RUST: Command channel closed");
                        break;
                    }
                }
            }
            
            // Handle incoming WebSocket messages
            message = ws_stream.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        messages_received += 1;
                        println!("📥 RUST: Received WebSocket text #{}: {}", messages_received, text);
                        
                        // Write all messages to log for debugging
                        use std::fs::OpenOptions;
                        use std::io::Write;
                        if let Ok(mut file) = OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open("/tmp/soniox_messages.log")
                        {
                            writeln!(file, "Message #{}: {}", messages_received, text).ok();
                        }
                        
                        // Check if this is just a keep-alive or processing update
                        if text.contains("\"tokens\":[]") && !text.contains("\"error\"") {
                            // Parse to check processing time
                            if let Ok(resp) = serde_json::from_str::<SonioxResponse>(&text) {
                                if let Some(proc_ms) = resp.total_audio_proc_ms {
                                    println!("💓 RUST: Processing update - total_audio_proc_ms: {}ms", proc_ms);
                                }
                            }
                            // Don't continue - send this as an empty result to keep the iterator alive
                            let result = StreamingResult {
                                session_id,
                                is_final: false,
                                text: String::new(),
                                confidence: 0.0,
                                timestamp: chrono::Utc::now(),
                                words: None,
                                metadata: Some(json!({"type": "processing_update"})),
                            };
                            let _ = result_tx.send(WebSocketMessage::Transcript(result));
                            continue;
                        }
                        
                        // Parse Soniox response
                        match serde_json::from_str::<SonioxResponse>(&text) {
                            Ok(response) => {
                                // Check for errors
                                if let Some(error) = response.error {
                                    let _ = result_tx.send(WebSocketMessage::Error(error));
                                    continue;
                                }
                                if let Some(error_msg) = response.error_message {
                                    let _ = result_tx.send(WebSocketMessage::Error(error_msg));
                                    continue;
                                }
                                
                                // Process tokens
                                if let Some(tokens) = response.tokens {
                                    if tokens.is_empty() {
                                        println!("💓 RUST: Keep-alive message (empty tokens)");
                                        continue;
                                    }
                                    
                                    // Build streaming result from tokens
                                    let mut full_text = String::new();
                                    let mut words = Vec::new();
                                    let mut confidence_sum = 0.0;
                                    let mut confidence_count = 0;
                                    let mut is_final = false;
                                    let mut detected_language = None;
                                    
                                    for token in tokens {
                                        full_text.push_str(&token.text);
                                        
                                        if let Some(conf) = token.confidence {
                                            confidence_sum += conf;
                                            confidence_count += 1;
                                        }
                                        
                                        if token.is_final {
                                            is_final = true;
                                        }
                                        
                                        // Extract language if available
                                        if let Some(lang) = &token.language {
                                            detected_language = Some(lang.clone());
                                        }
                                        
                                        // Create word timing if available
                                        if let (Some(start), Some(duration)) = (token.start_time, token.duration) {
                                            words.push(WordTiming {
                                                word: token.text.clone(),
                                                start,
                                                end: start + duration,
                                                confidence: token.confidence.unwrap_or(1.0),
                                            });
                                        }
                                    }
                                    
                                    let avg_confidence = if confidence_count > 0 {
                                        confidence_sum / confidence_count as f32
                                    } else {
                                        1.0
                                    };
                                    
                                    let result = StreamingResult {
                                        session_id,
                                        is_final,
                                        text: full_text,
                                        confidence: avg_confidence,
                                        timestamp: chrono::Utc::now(),
                                        words: if words.is_empty() { None } else { Some(words) },
                                        metadata: detected_language.map(|lang| {
                                            json!({"detected_language": lang})
                                        }),
                                    };
                                    
                                    println!("🎯 RUST: Sending transcript result: '{}' (final={})", result.text, result.is_final);
                                    
                                    // Also write to a log file for debugging
                                    if !result.text.is_empty() {
                                        use std::fs::OpenOptions;
                                        use std::io::Write;
                                        if let Ok(mut file) = OpenOptions::new()
                                            .create(true)
                                            .append(true)
                                            .open("/tmp/soniox_transcription.log")
                                        {
                                            writeln!(file, "Transcription: '{}'", result.text).ok();
                                        }
                                    }
                                    
                                    let _ = result_tx.send(WebSocketMessage::Transcript(result));
                                }
                            }
                            Err(e) => {
                                println!("❌ RUST: Failed to parse response: {}", e);
                                let _ = result_tx.send(WebSocketMessage::Error(format!("Parse error: {}", e)));
                            }
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        println!("🏓 RUST: Received ping, sending pong");
                        if let Err(e) = ws_stream.send(Message::Pong(data)).await {
                            let _ = result_tx.send(WebSocketMessage::Error(format!("Pong failed: {}", e)));
                            break;
                        }
                    }
                    Some(Ok(Message::Pong(_))) => {
                        println!("🏓 RUST: Received pong");
                    }
                    Some(Ok(Message::Close(_))) => {
                        println!("🛑 RUST: WebSocket closed by server");
                        let _ = result_tx.send(WebSocketMessage::Closed);
                        break;
                    }
                    Some(Ok(msg)) => {
                        println!("🔍 RUST: Other message type: {:?}", msg);
                    }
                    Some(Err(e)) => {
                        println!("❌ RUST: WebSocket error: {}", e);
                        let _ = result_tx.send(WebSocketMessage::Error(format!("WebSocket error: {}", e)));
                        break;
                    }
                    None => {
                        println!("📪 RUST: WebSocket stream ended");
                        let _ = result_tx.send(WebSocketMessage::Closed);
                        break;
                    }
                }
            }
        }
    }
    
    println!("🏁 RUST: WebSocket handler exiting for session {}", session_id);
}

#[derive(Debug, Deserialize)]
struct SonioxResponse {
    tokens: Option<Vec<SonioxToken>>,
    final_audio_proc_ms: Option<i32>,
    total_audio_proc_ms: Option<i32>,
    error: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SonioxToken {
    text: String,
    #[serde(default)]
    is_final: bool,
    confidence: Option<f32>,
    language: Option<String>,
    start_time: Option<f32>,
    duration: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_provider_creation() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map
        });
        
        let provider = SonioxProvider::new(&config).await;
        assert!(provider.is_ok());
    }
    
    #[tokio::test]
    async fn test_auto_detect_language_config() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("auto_detect_language".to_string(), json!(true));
            map
        });
        
        let provider = SonioxProvider::new(&config).await.unwrap();
        assert!(provider.auto_detect_language);
    }
    
    #[test]
    fn test_supported_formats() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let formats = provider.supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.iter().any(|f| f.format == "wav"));
    }
    
    #[tokio::test]
    async fn test_list_models_includes_auto() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: true,
        };
        
        let models = provider.list_models().await.unwrap();
        assert!(models.iter().any(|m| m.id == "auto"));
        assert!(models.iter().any(|m| m.capabilities.contains(&"language-detection".to_string())));
    }

    #[test]
    fn test_provider_config_api_key_extraction() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("my-secret-key"));
            map.insert("model".to_string(), json!("es"));
            map.insert("auto_detect_language".to_string(), json!(false));
            map
        });
        
        assert_eq!(config.get_api_key(), Some("my-secret-key".to_string()));
        assert_eq!(config.get_value("model").unwrap().as_str(), Some("es"));
        assert_eq!(config.get_value("auto_detect_language").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn test_provider_config_missing_api_key() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("model".to_string(), json!("en"));
            map
        });
        
        assert!(config.get_api_key().is_none());
    }

    #[tokio::test]
    async fn test_provider_creation_fails_without_api_key() {
        let config = ProviderConfig::Simple(std::collections::HashMap::new());
        
        let result = SonioxProvider::new(&config).await;
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(matches!(error, DebabelizerError::Configuration(_)));
    }

    #[tokio::test]
    async fn test_provider_creation_with_custom_model() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("model".to_string(), json!("fr"));
            map
        });
        
        let provider = SonioxProvider::new(&config).await.unwrap();
        assert_eq!(provider.model, "fr");
    }

    #[test]
    fn test_provider_name() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        assert_eq!(provider.name(), "soniox");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supported_formats_contains_multiple_sample_rates() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let formats = provider.supported_formats();
        assert!(formats.len() >= 2);
        
        let sample_rates: Vec<u32> = formats.iter().map(|f| f.sample_rate).collect();
        assert!(sample_rates.contains(&16000));
        assert!(sample_rates.contains(&48000));
    }

    #[tokio::test]
    async fn test_list_models_without_auto_detect() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let models = provider.list_models().await.unwrap();
        assert!(models.iter().any(|m| m.id == "en"));
        assert!(models.iter().any(|m| m.id == "es"));
        assert!(models.iter().any(|m| m.id == "fr"));
        assert!(models.iter().any(|m| m.id == "de"));
        assert!(models.iter().any(|m| m.id == "hi"));
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        
        assert!(config.language.is_none());
        assert!(config.model.is_none());
        assert!(config.interim_results); // Default is true
        assert!(config.punctuate); // Default is true
        assert!(!config.profanity_filter); // Default is false
    }

    #[test]
    fn test_stream_config_custom() {
        let config = StreamConfig {
            session_id: uuid::Uuid::new_v4(),
            language: Some("en-US".to_string()),
            model: Some("en".to_string()),
            format: AudioFormat::wav(16000),
            interim_results: true,
            punctuate: true,
            profanity_filter: false,
            diarization: true,
            metadata: None,
            enable_word_time_offsets: false,
            enable_automatic_punctuation: false,
            enable_language_identification: false,
        };
        
        assert_eq!(config.language, Some("en-US".to_string()));
        assert_eq!(config.model, Some("en".to_string()));
        assert!(config.interim_results);
        assert!(config.punctuate);
        assert!(config.diarization);
    }

    #[test]
    fn test_soniox_response_parsing() {
        let json_response = r#"{
            "tokens": [
                {
                    "text": "Hello",
                    "is_final": false,
                    "confidence": 0.98,
                    "start_time": 0.0,
                    "duration": 0.5
                },
                {
                    "text": "world",
                    "is_final": true,
                    "confidence": 0.92,
                    "start_time": 0.6,
                    "duration": 0.4
                }
            ],
            "final_audio_proc_ms": 100,
            "total_audio_proc_ms": 500
        }"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        let tokens = response.tokens.unwrap();
        
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert!(!tokens[0].is_final);
        assert_eq!(tokens[0].confidence, Some(0.98));
        assert_eq!(tokens[0].start_time, Some(0.0));
        assert_eq!(tokens[0].duration, Some(0.5));
        
        assert_eq!(tokens[1].text, "world");
        assert!(tokens[1].is_final);
        assert_eq!(tokens[1].confidence, Some(0.92));
    }

    #[test]
    fn test_soniox_response_minimal() {
        let json_response = r#"{
            "tokens": [
                {
                    "text": "Test",
                    "is_final": false
                }
            ]
        }"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        let tokens = response.tokens.unwrap();
        
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Test");
        assert!(!tokens[0].is_final);
        assert!(tokens[0].confidence.is_none());
        assert!(tokens[0].language.is_none());
        assert!(tokens[0].start_time.is_none());
    }

    #[test]
    fn test_soniox_response_empty() {
        let json_response = r#"{"tokens":[]}"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        let tokens = response.tokens.unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_websocket_url_construction() {
        let _provider = SonioxProvider {
            api_key: "test-key".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        // Test URL construction logic - updated to correct endpoint
        let base_url = "wss://stt-rt.soniox.com/transcribe-websocket";
        assert_eq!(SONIOX_WS_URL, base_url);
    }

    #[test]
    fn test_word_timing_conversion() {
        let soniox_token = SonioxToken {
            text: "hello".to_string(),
            is_final: false,
            confidence: Some(0.95),
            language: None,
            start_time: Some(1.0),
            duration: Some(0.5),
        };
        
        let word_timing = WordTiming {
            word: soniox_token.text.clone(),
            start: soniox_token.start_time.unwrap(),
            end: soniox_token.start_time.unwrap() + soniox_token.duration.unwrap(),
            confidence: soniox_token.confidence.unwrap_or(1.0),
        };
        
        assert_eq!(word_timing.word, "hello");
        assert_eq!(word_timing.start, 1.0);
        assert_eq!(word_timing.end, 1.5);
        assert_eq!(word_timing.confidence, 0.95);
    }

    #[test]
    fn test_auto_detect_language_provider() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "auto".to_string(),
            auto_detect_language: true,
        };
        
        assert_eq!(provider.model, "auto");
        assert!(provider.auto_detect_language);
    }

    #[test]
    fn test_provider_constants() {
        assert_eq!(SONIOX_WS_URL, "wss://stt-rt.soniox.com/transcribe-websocket");
    }

    // Test streaming result creation
    #[test]
    fn test_streaming_result_creation() {
        let session_id = uuid::Uuid::new_v4();
        let words = vec![
            WordTiming {
                word: "hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.98,
            },
            WordTiming {
                word: "world".to_string(),
                start: 0.6,
                end: 1.0,
                confidence: 0.92,
            },
        ];
        
        let mut result = StreamingResult::new(session_id, "hello world".to_string(), true, 0.95);
        result.words = Some(words.clone());
        
        assert_eq!(result.session_id, session_id);
        assert!(result.is_final);
        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.words.unwrap().len(), 2);
    }

    #[test]
    fn test_model_info_creation() {
        let model = Model {
            id: "soniox-en".to_string(),
            name: "Soniox English Model".to_string(),
            languages: vec!["en".to_string()],
            capabilities: vec!["transcription".to_string(), "streaming".to_string()],
        };
        
        assert_eq!(model.id, "soniox-en");
        assert_eq!(model.name, "Soniox English Model");
        assert_eq!(model.languages.len(), 1);
        assert_eq!(model.capabilities.len(), 2);
    }
}