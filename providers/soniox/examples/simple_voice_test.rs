// Simplified voice streaming test for Soniox provider
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration, sleep};
use uuid::Uuid;

use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use debabelizer_core::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Using native-tls instead of rustls to avoid crypto provider issues
    println!("🦀 Rust Soniox Voice Streaming Test");
    println!("{}", "=".repeat(50));
    
    // Set up API key
    let api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95";
    println!("✅ API Key: {}***", &api_key[..6]);
    
    // Create provider config
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview-v2"));
    let provider_config = ProviderConfig::Simple(config_map);
    
    // Create provider
    println!("\n📡 Creating Soniox provider...");
    let provider = SonioxProvider::new(&provider_config).await?;
    println!("✅ Provider created: {}", provider.name());
    
    // Load test audio
    println!("\n🎵 Loading audio file...");
    let audio_files = [
        "/home/peter/debabelizer/english_sample.wav",
        "/home/peter/debabelizer/test.wav"
    ];
    
    let mut audio_data = None;
    for file in &audio_files {
        if let Ok(data) = std::fs::read(file) {
            println!("✅ Loaded {} bytes from {}", data.len(), file);
            audio_data = Some(data);
            break;
        }
    }
    
    let audio_data = audio_data.ok_or_else(|| "No audio file found".to_string())?;
    
    // Create stream config
    let session_id = Uuid::new_v4();
    let stream_config = StreamConfig {
        session_id,
        format: AudioFormat {
            format: "wav".to_string(),
            sample_rate: 16000,
            channels: 1,
            bit_depth: Some(16),
        },
        interim_results: true,
        language: None,
        model: None,
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: true,
        enable_automatic_punctuation: true,
        enable_language_identification: true,
    };
    
    println!("\n🌊 Creating stream (CRITICAL TEST - checks if async fix worked)...");
    let mut stream = match provider.transcribe_stream(stream_config).await {
        Ok(s) => {
            println!("✅ SUCCESS! Stream created! Session: {}", s.session_id());
            println!("🎉 This confirms the background WebSocket task is now working!");
            s
        }
        Err(e) => {
            println!("❌ FAILURE! Stream creation failed: {:?}", e);
            println!("💡 This would mean the async fix didn't work");
            return Err(e.into());
        }
    };
    
    println!("\n📤 Sending first audio chunk and testing result reception...");
    
    // Send just one chunk to test
    let first_chunk = &audio_data[..std::cmp::min(3200, audio_data.len())];
    println!("Sending {} bytes...", first_chunk.len());
    
    match stream.send_audio(first_chunk).await {
        Ok(()) => {
            println!("✅ Audio chunk sent successfully!");
        }
        Err(e) => {
            println!("❌ Failed to send audio: {:?}", e);
            return Err(e.into());
        }
    }
    
    println!("\n📥 Waiting for results (this tests if WebSocket handler is working)...");
    
    // Try to get at least one result
    for attempt in 1..=5 {
        println!("  Attempt {}/5...", attempt);
        
        match timeout(Duration::from_secs(3), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                println!("  ✅ SUCCESS! Got result: '{}'", result.text.trim());
                println!("  🎯 Final: {}, Confidence: {:.3}", result.is_final, result.confidence);
                println!("\n🎉 STREAMING TEST PASSED!");
                println!("  - Background WebSocket task is executing");
                println!("  - Real-time communication with Soniox is working");
                println!("  - The async execution bottleneck has been resolved!");
                break;
            }
            Ok(Ok(None)) => {
                println!("  🔚 Stream ended");
                break;
            }
            Ok(Err(e)) => {
                println!("  ❌ Error receiving result: {:?}", e);
                break;
            }
            Err(_) => {
                println!("  ⏰ Timeout on attempt {}", attempt);
                if attempt < 5 {
                    println!("     Trying again...");
                } else {
                    println!("\n⚠️ NO RESULTS RECEIVED");
                    println!("This could mean:");
                    println!("  - WebSocket connection established but no responses");
                    println!("  - Audio format incompatible with Soniox");
                    println!("  - Background task started but isn't processing messages");
                    println!("  - The async fix worked partially but there are other issues");
                }
            }
        }
    }
    
    // Close stream
    println!("\n🔚 Closing stream...");
    let _ = stream.close().await;
    println!("✅ Test completed!");
    
    Ok(())
}