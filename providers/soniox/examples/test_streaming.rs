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
    
    println!("🚀 Testing Soniox streaming transcription directly in Rust...");
    
    // Load API key from environment
    let api_key = std::env::var("SONIOX_API_KEY")
        .expect("SONIOX_API_KEY environment variable not set");
    
    // Create provider config
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview"));
    
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
    
    // Create stream config
    let stream_config = StreamConfig {
        session_id: Uuid::new_v4(),
        language: None, // Use auto-detection
        model: None,
        format: audio_format,
        interim_results: true,
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: false,
        enable_automatic_punctuation: true,
        enable_language_identification: true, // Enable auto-detection
    };
    
    // Start streaming
    println!("\n🎤 Creating streaming session...");
    let mut stream = provider.transcribe_stream(stream_config).await?;
    println!("✅ Stream created");
    
    // Send audio in chunks
    let chunk_size = 3200; // 200ms at 16kHz
    let chunks: Vec<_> = pcm_data.chunks(chunk_size).collect();
    
    println!("\n📊 Streaming {} chunks of audio...", chunks.len());
    
    // Send audio chunks first
    for (i, chunk) in chunks.iter().enumerate() {
        println!("📤 Sending chunk {}/{}: {} bytes", i + 1, chunks.len(), chunk.len());
        stream.send_audio(chunk).await?;
        sleep(Duration::from_millis(200)).await; // Simulate real-time
    }
    
    println!("\n⏳ Collecting results...");
    
    // Collect results
    let mut results = Vec::new();
    let mut count = 0;
    let start_time = std::time::Instant::now();
    
    loop {
        match stream.receive_transcript().await {
            Ok(Some(result)) => {
                count += 1;
                if !result.text.is_empty() {
                    println!("\n🎯 Result {}: '{}' (final={})", count, result.text, result.is_final);
                    results.push(result.text);
                } else {
                    println!("💓 Keep-alive {}", count);
                }
                
                // Check if we've received results for a while
                if start_time.elapsed().as_secs() > 10 {
                    println!("\n⏰ Timeout after 10 seconds");
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
    }
    
    // Close stream
    println!("🛑 Closing stream...");
    stream.close().await?;
    
    println!("\n📊 Summary:");
    println!("- Total transcriptions: {}", results.len());
    if !results.is_empty() {
        println!("- Full text: {}", results.join(" "));
    } else {
        println!("- No transcription received ❌");
    }
    
    Ok(())
}