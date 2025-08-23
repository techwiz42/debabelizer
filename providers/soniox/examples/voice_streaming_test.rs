// Simple voice streaming test for Soniox provider
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration, sleep};
use uuid::Uuid;

use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use debabelizer_core::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    
    let audio_data = audio_data.ok_or("No audio file found")?;
    
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
    
    println!("\n🌊 Creating stream (testing async fix)...");
    let mut stream = match provider.transcribe_stream(stream_config).await {
        Ok(s) => {
            println!("✅ Stream created! Session: {}", s.session_id());
            s
        }
        Err(e) => {
            println!("❌ Stream creation failed: {:?}", e);
            return Err(e.into());
        }
    };
    
    println!("\n📤 Sending audio and collecting results...");
    
    // Send audio in chunks
    let chunk_size = 3200;
    let chunks: Vec<_> = audio_data.chunks(chunk_size).collect();
    println!("Sending {} chunks...", chunks.len());
    
    // Start sending and receiving concurrently
    let send_task = async {
        for (i, chunk) in chunks.iter().enumerate() {
            println!("  📤 Chunk {}/{}", i + 1, chunks.len());
            stream.send_audio(chunk).await?;
            sleep(Duration::from_millis(100)).await;
        }
        println!("✅ All chunks sent");
        Ok::<(), debabelizer_core::DebabelizerError>(())
    };
    
    let receive_task = async {
        let mut results = Vec::new();
        for i in 0..20 { // Max 20 results
            match timeout(Duration::from_secs(3), stream.receive_transcript()).await {
                Ok(Ok(Some(result))) => {
                    println!("  📥 {}: '{}' (final: {})", i + 1, result.text.trim(), result.is_final);
                    results.push(result);
                }
                Ok(Ok(None)) => {
                    println!("  🔚 Stream ended");
                    break;
                }
                Ok(Err(e)) => {
                    println!("  ❌ Error: {:?}", e);
                    break;
                }
                Err(_) => {
                    println!("  ⏰ Timeout");
                    if results.is_empty() {
                        continue;
                    } else {
                        break;
                    }
                }
            }
        }
        Ok::<Vec<StreamingResult>, debabelizer_core::DebabelizerError>(results)
    };
    
    // Run both tasks
    match timeout(Duration::from_secs(30), tokio::try_join!(send_task, receive_task)).await {
        Ok(Ok((_, results))) => {
            println!("\n📊 Results: {} total", results.len());
            
            if results.is_empty() {
                println!("❌ No results - WebSocket task may not be executing");
            } else {
                println!("✅ Got results - async fix worked!");
                
                // Show final transcript
                let final_text: String = results
                    .iter()
                    .filter(|r| r.is_final)
                    .map(|r| r.text.trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                    
                if !final_text.is_empty() {
                    println!("📝 Transcript: \"{}\"", final_text);
                } else {
                    println!("📝 No final transcript (only interim results)");
                }
            }
        }
        Ok(Err(e)) => {
            println!("❌ Task error: {:?}", e);
        }
        Err(_) => {
            println!("⏰ Test timed out");
        }
    }
    
    // Close stream
    let _ = stream.close().await;
    println!("\n🎉 Test completed!");
    
    Ok(())
}