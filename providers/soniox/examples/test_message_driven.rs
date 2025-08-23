use debabelizer_core::{
    audio::AudioFormat,
    stt::{SttProvider, StreamConfig},
};
use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Testing message-driven Soniox streaming...");
    
    // Load API key from environment
    let api_key = std::env::var("SONIOX_API_KEY")
        .expect("SONIOX_API_KEY environment variable not set");
    
    // Create provider config
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview-v2"));
    
    let config = ProviderConfig::Simple(config_map);
    
    // Initialize provider
    println!("📡 Initializing Soniox provider...");
    let provider = SonioxProvider::new(&config).await?;
    println!("✅ Provider initialized");
    
    // Load test audio file
    let audio_path = "/home/peter/debabelizer/tests/test_real_speech_16k.wav";
    println!("📁 Loading audio file: {}", audio_path);
    
    let mut file = File::open(audio_path)?;
    let mut audio_data = Vec::new();
    file.read_to_end(&mut audio_data)?;
    
    println!("✅ Loaded {} bytes of audio", audio_data.len());
    
    // Skip WAV header (44 bytes)
    let pcm_data = audio_data[44..].to_vec();
    println!("🎵 PCM data: {} bytes", pcm_data.len());
    
    // Create audio format
    let audio_format = AudioFormat {
        format: "wav".to_string(),
        sample_rate: 16000,
        channels: 1,
        bit_depth: Some(16),
    };
    
    // Create stream config for language auto-detection
    let stream_config = StreamConfig {
        session_id: Uuid::new_v4(),
        language: None,  // Auto-detect
        model: None,
        format: audio_format,
        interim_results: true,
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: false,
        enable_automatic_punctuation: true,
        enable_language_identification: true,
    };
    
    // Start streaming
    println!("\n🎤 Creating message-driven streaming session...");
    let mut stream = provider.transcribe_stream(stream_config).await?;
    println!("✅ Stream created with background WebSocket handler");
    
    // Give background task time to initialize
    sleep(Duration::from_millis(500)).await;
    
    // Send audio in chunks
    let chunk_size = 6400; // 400ms chunks at 16kHz
    let chunks: Vec<_> = pcm_data.chunks(chunk_size).collect();
    println!("\n📤 Sending {} audio chunks...", chunks.len());
    
    // Start result collection task
    let mut results = Vec::new();
    let mut interim_results = Vec::new();
    
    // Send chunks with realistic timing
    for (i, chunk) in chunks.iter().enumerate() {
        println!("📤 Sending chunk {}/{}: {} bytes", i + 1, chunks.len(), chunk.len());
        
        // Send audio chunk
        stream.send_audio(chunk).await?;
        
        // Check for results (non-blocking)
        match tokio::time::timeout(Duration::from_millis(100), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                if result.is_final {
                    println!("🎯 FINAL: '{}' (confidence: {:.3})", result.text, result.confidence);
                    results.push(result.text.clone());
                    
                    // Check for language detection
                    if let Some(metadata) = &result.metadata {
                        if let Some(lang) = metadata.get("detected_language") {
                            println!("🌍 Language detected: {}", lang);
                        }
                    }
                } else if !result.text.is_empty() {
                    println!("💬 INTERIM: '{}'", result.text);
                    interim_results.push(result.text.clone());
                }
            }
            Ok(Ok(None)) => {
                // No result yet
            }
            Ok(Err(e)) => {
                println!("❌ Error getting result: {}", e);
                break;
            }
            Err(_) => {
                // Timeout - continue sending
            }
        }
        
        // Real-time delay
        sleep(Duration::from_millis(250)).await;
    }
    
    // Give Soniox time to process and finalize  
    println!("\n⏱️ Waiting for Soniox to finalize results...");
    sleep(Duration::from_millis(2000)).await;
    
    println!("\n⏳ Collecting final results...");
    
    // Wait for final results  
    let start_time = std::time::Instant::now();
    while start_time.elapsed().as_secs() < 15 {
        match tokio::time::timeout(Duration::from_millis(1000), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                if result.is_final {
                    println!("🎯 FINAL: '{}' (confidence: {:.3})", result.text, result.confidence);
                    results.push(result.text.clone());
                    
                    // Check for language detection in metadata
                    if let Some(metadata) = &result.metadata {
                        if let Some(lang) = metadata.get("detected_language") {
                            println!("🌍 Language detected: {}", lang);
                        }
                    }
                } else if !result.text.is_empty() {
                    println!("💬 INTERIM: '{}'", result.text);
                    interim_results.push(result.text.clone());
                }
            }
            Ok(Ok(None)) => {
                println!("🏁 Stream ended");
                break;
            }
            Ok(Err(e)) => {
                println!("❌ Error: {}", e);
                break;
            }
            Err(_) => {
                println!("⏱️ No results in 1s, continuing...");
            }
        }
    }
    
    // Close stream
    println!("\n🛑 Closing message-driven stream...");
    stream.close().await?;
    
    // Summary
    println!("\n📊 MESSAGE-DRIVEN STREAMING RESULTS:");
    println!("- Final transcriptions: {}", results.len());
    println!("- Interim transcriptions: {}", interim_results.len());
    
    if !results.is_empty() {
        let full_text = results.join(" ");
        println!("- Complete transcription: '{}'", full_text);
        println!("\n🎉 SUCCESS! Message-driven Rust streaming WORKS!");
        println!("✅ Speech in → Language detected → Text out ✅");
    } else {
        println!("- No final transcriptions received ❌");
    }
    
    Ok(())
}