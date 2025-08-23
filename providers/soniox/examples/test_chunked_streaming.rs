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
    
    println!("🚀 Testing Soniox with chunked streaming...");
    
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
    let audio_path = "/home/peter/debabelizer/test_real_speech_16k.wav";
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
    
    // Create stream config for auto-detection
    let stream_config = StreamConfig {
        session_id: Uuid::new_v4(),
        language: None,  // Auto-detect language
        model: None,
        format: audio_format,
        interim_results: true,
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: false,
        enable_automatic_punctuation: true,
        enable_language_identification: true,  // Enable auto-detection
    };
    
    // Start streaming
    println!("\n🎤 Creating streaming session...");
    let mut stream = provider.transcribe_stream(stream_config).await?;
    println!("✅ Stream created");
    
    // Send audio in smaller chunks (like Python implementation)
    let chunk_size = 8192; // 8KB chunks like Python
    let chunks: Vec<_> = pcm_data.chunks(chunk_size).collect();
    println!("\n📤 Sending {} chunks of audio...", chunks.len());
    
    // Start result collection in background
    let mut results = Vec::new();
    
    // Send chunks with delays 
    for (i, chunk) in chunks.iter().enumerate() {
        println!("📤 Sending chunk {}/{}: {} bytes", i + 1, chunks.len(), chunk.len());
        stream.send_audio(chunk).await?;
        
        // Check for results after each chunk
        if let Ok(Some(result)) = stream.receive_transcript().await {
            if !result.text.is_empty() {
                println!("🎯 Result after chunk {}: '{}' (final={})", i + 1, result.text, result.is_final);
                results.push(result.text.clone());
            } else {
                println!("💓 Keep-alive after chunk {}", i + 1);
            }
        }
        
        // Small delay between chunks for real-time simulation
        sleep(Duration::from_millis(100)).await;
    }
    
    println!("\n⏳ Collecting final results...");
    
    // Collect remaining results
    let start_time = std::time::Instant::now();
    loop {
        match stream.receive_transcript().await {
            Ok(Some(result)) => {
                if !result.text.is_empty() {
                    println!("🎯 Final result: '{}' (final={})", result.text, result.is_final);
                    results.push(result.text);
                }
                
                if result.is_final {
                    println!("✅ Got final result!");
                    break;
                }
            }
            Ok(None) => {
                println!("🏁 Stream ended");
                break;
            }
            Err(e) => {
                println!("❌ Error receiving result: {}", e);
                break;
            }
        }
        
        // Timeout after 10 seconds
        if start_time.elapsed().as_secs() > 10 {
            println!("⏰ Timeout waiting for results");
            break;
        }
    }
    
    // Close stream
    println!("🛑 Closing stream...");
    stream.close().await?;
    
    println!("\n📊 Summary:");
    println!("- Total transcriptions: {}", results.len());
    if !results.is_empty() {
        let full_text = results.join(" ");
        println!("- Full transcription: '{}'", full_text);
        println!("🎉 SUCCESS: Rust Soniox streaming WORKS!");
    } else {
        println!("- No transcription received ❌");
    }
    
    Ok(())
}