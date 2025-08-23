use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

// Import the Soniox provider and debabelizer core
use debabelizer_core::*;

#[path = "providers/soniox/src/lib.rs"]
mod soniox_provider;

use soniox_provider::{SonioxProvider, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for better debugging
    tracing_subscriber::init();
    
    println!("🦀 Testing Pure Rust Soniox Streaming STT Implementation");
    println!("{}", "=".repeat(60));
    
    // Check if we have a Soniox API key
    let api_key = std::env::var("SONIOX_API_KEY")
        .or_else(|_| std::env::var("SONIOX_KEY"))
        .expect("❌ Please set SONIOX_API_KEY environment variable");
    
    println!("✅ Found Soniox API key: {}***", &api_key[..std::cmp::min(6, api_key.len())]);
    
    // Create provider configuration
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview-v2"));
    config_map.insert("auto_detect_language".to_string(), json!(true));
    
    let provider_config = ProviderConfig::Simple(config_map);
    
    // Create Soniox provider
    println!("\n📡 Creating Soniox STT Provider...");
    let provider = match SonioxProvider::new(&provider_config).await {
        Ok(p) => {
            println!("✅ Soniox provider created successfully");
            p
        },
        Err(e) => {
            println!("❌ Failed to create provider: {:?}", e);
            return Err(e.into());
        }
    };
    
    // Test 1: Provider capabilities
    println!("\n🔍 Testing Provider Capabilities:");
    println!("  - Name: {}", provider.name());
    println!("  - Supports streaming: {}", provider.supports_streaming());
    println!("  - Supported formats: {:?}", provider.supported_formats());
    
    let models = provider.list_models().await?;
    println!("  - Available models: {} models", models.len());
    for model in &models[..3] { // Show first 3 models
        println!("    * {} ({})", model.name, model.id);
    }
    
    // Test 2: Load test audio file
    println!("\n🎵 Loading test audio file...");
    let test_file = "/home/peter/debabelizer/english_sample.wav";
    let audio_data = match std::fs::read(test_file) {
        Ok(data) => {
            println!("✅ Loaded {} bytes from {}", data.len(), test_file);
            data
        },
        Err(e) => {
            println!("❌ Failed to load audio file: {}", e);
            println!("ℹ️ Trying alternative: test.wav");
            std::fs::read("/home/peter/debabelizer/test.wav")
                .expect("Failed to load backup test.wav file")
        }
    };
    
    // Create AudioData with proper format
    let audio_format = AudioFormat {
        format: "wav".to_string(),
        sample_rate: 16000,
        channels: 1,
        bit_depth: Some(16),
    };
    
    let audio = AudioData {
        data: audio_data,
        format: audio_format,
    };
    
    // Test 3: Create streaming session
    println!("\n🌊 Testing Streaming Transcription:");
    let session_id = Uuid::new_v4();
    let stream_config = StreamConfig {
        session_id,
        format: audio.format.clone(),
        interim_results: true,
        language: None, // Auto-detect
        model: None,
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: true,
        enable_automatic_punctuation: true,
        enable_language_identification: true,
    };
    
    println!("  📋 Session ID: {}", session_id);
    println!("  🎧 Audio format: {} Hz, {} channels", stream_config.format.sample_rate, stream_config.format.channels);
    println!("  🔄 Interim results: {}", stream_config.interim_results);
    
    // Create streaming connection
    println!("\n🔗 Creating streaming connection to Soniox...");
    let mut stream = match provider.transcribe_stream(stream_config).await {
        Ok(s) => {
            println!("✅ Streaming connection established");
            s
        },
        Err(e) => {
            println!("❌ Failed to create stream: {:?}", e);
            return Err(e.into());
        }
    };
    
    // Test 4: Send audio in chunks and collect results
    println!("\n📤 Sending audio data in chunks...");
    let chunk_size = 3200; // Send in small chunks to simulate streaming
    let total_chunks = (audio.data.len() + chunk_size - 1) / chunk_size;
    println!("  📊 Total audio: {} bytes, {} chunks of {} bytes", audio.data.len(), total_chunks, chunk_size);
    
    // Start a background task to collect results
    let mut results = Vec::new();
    let collect_results = async {
        let mut result_count = 0;
        println!("  🎯 Starting result collection...");
        
        while let Some(result) = stream.receive_transcript().await? {
            result_count += 1;
            println!("  📥 Result #{}: '{}' (final: {}, confidence: {:.2})", 
                result_count, 
                result.text.trim(), 
                result.is_final, 
                result.confidence
            );
            
            // Check metadata
            if let Some(metadata) = &result.metadata {
                if let Some(msg_type) = metadata.get("type") {
                    println!("    ℹ️ Metadata type: {}", msg_type);
                }
            }
            
            // Store non-empty final results
            if result.is_final && !result.text.trim().is_empty() {
                results.push(result);
            }
            
            // Stop after collecting reasonable number of results or timeout
            if result_count >= 20 {
                println!("  ⏹️ Stopping after {} results", result_count);
                break;
            }
        }
        
        println!("  ✅ Result collection finished with {} total results", result_count);
        Ok::<(), debabelizer_core::DebabelizerError>(())
    };
    
    // Send audio chunks
    let send_audio = async {
        for (i, chunk) in audio.data.chunks(chunk_size).enumerate() {
            println!("  📤 Sending chunk {}/{} ({} bytes)", i + 1, total_chunks, chunk.len());
            
            if let Err(e) = stream.send_audio(chunk).await {
                println!("  ❌ Failed to send chunk {}: {:?}", i + 1, e);
                return Err(e);
            }
            
            // Small delay between chunks to simulate real-time streaming
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        
        println!("  ✅ All audio chunks sent successfully");
        
        // Give a moment for final processing
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Close the stream
        println!("  🔚 Closing stream...");
        stream.close().await?;
        println!("  ✅ Stream closed");
        
        Ok(())
    };
    
    // Run both tasks concurrently with timeout
    println!("\n⏰ Running streaming test with 30-second timeout...");
    match timeout(Duration::from_secs(30), tokio::try_join!(send_audio, collect_results)).await {
        Ok(Ok(((), ()))) => {
            println!("✅ Streaming test completed successfully");
        },
        Ok(Err(e)) => {
            println!("❌ Streaming test failed: {:?}", e);
            return Err(e.into());
        },
        Err(_) => {
            println!("⏰ Streaming test timed out after 30 seconds");
            return Err("Test timeout".into());
        }
    }
    
    // Test 5: Analyze results
    println!("\n📊 Results Analysis:");
    println!("  📋 Total final results: {}", results.len());
    
    if results.is_empty() {
        println!("  ⚠️ No final transcription results received");
    } else {
        // Combine all final text
        let full_transcript: String = results.iter()
            .map(|r| r.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<&str>>()
            .join(" ");
        
        println!("  📝 Full transcript: '{}'", full_transcript);
        println!("  📏 Transcript length: {} characters", full_transcript.len());
        
        // Calculate average confidence
        let avg_confidence: f32 = results.iter()
            .map(|r| r.confidence)
            .sum::<f32>() / results.len() as f32;
        println!("  🎯 Average confidence: {:.3}", avg_confidence);
        
        // Show word timing info if available
        let total_words: usize = results.iter()
            .filter_map(|r| r.words.as_ref())
            .map(|words| words.len())
            .sum();
        if total_words > 0 {
            println!("  ⏰ Total words with timing: {}", total_words);
        }
    }
    
    println!("\n🎉 Pure Rust Soniox streaming test completed!");
    println!("{}", "=".repeat(60));
    
    Ok(())
}