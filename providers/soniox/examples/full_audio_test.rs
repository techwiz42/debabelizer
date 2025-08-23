// Full audio streaming test for Soniox provider
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration, sleep};
use uuid::Uuid;

use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use debabelizer_core::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Rust Soniox Full Audio Streaming Test");
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
    
    // Skip WAV header (44 bytes) to get raw PCM data
    let pcm_data = if audio_data.len() > 44 {
        &audio_data[44..]
    } else {
        &audio_data[..]
    };
    
    println!("📊 Audio info: {} total bytes, {} PCM bytes", audio_data.len(), pcm_data.len());
    
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
    
    println!("\n🌊 Creating stream...");
    let mut stream = provider.transcribe_stream(stream_config).await?;
    println!("✅ Stream created! Session: {}", stream.session_id());
    
    // Send audio in chunks (Python uses 8192 bytes)
    let chunk_size = 8192;
    let chunks: Vec<_> = pcm_data.chunks(chunk_size).collect();
    println!("\n📤 Sending {} chunks of audio...", chunks.len());
    
    // Send all audio chunks first
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  📤 Sending chunk {}/{} ({} bytes)", i + 1, chunks.len(), chunk.len());
        stream.send_audio(chunk).await?;
        
        // Small delay between chunks to simulate real-time audio
        sleep(Duration::from_millis(50)).await;
    }
    
    println!("\n✅ All audio sent! Collecting results...");
    
    // Now collect results
    let mut results = Vec::new();
    let mut result_count = 0;
    let mut no_result_count = 0;
    
    loop {
        match timeout(Duration::from_secs(2), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                result_count += 1;
                no_result_count = 0; // Reset timeout counter
                let text = result.text.trim();
                if !text.is_empty() || result.is_final {
                    println!("  📥 Result #{}: '{}' (final: {}, conf: {:.3})", 
                        result_count, text, result.is_final, result.confidence);
                    results.push(result);
                }
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
                no_result_count += 1;
                if no_result_count >= 3 {
                    if result_count == 0 {
                        println!("  ⏰ No results after 6 seconds");
                    } else {
                        println!("  ⏰ No more results after 6 seconds");
                    }
                    break;
                }
            }
        }
    }
    
    // Display final results
    println!("\n📊 Final Results:");
    println!("  - Total results: {}", results.len());
    
    let final_text: String = results
        .iter()
        .filter(|r| r.is_final)
        .map(|r| r.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    
    if !final_text.is_empty() {
        println!("\n📝 FINAL TRANSCRIPT: \"{}\"", final_text);
        println!("\n🎉 SUCCESS! Speech → Text conversion WORKING!");
    } else if !results.is_empty() {
        let interim_text: String = results
            .iter()
            .map(|r| r.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        
        if !interim_text.is_empty() {
            println!("\n📝 INTERIM TRANSCRIPT: \"{}\"", interim_text);
            println!("\n⚠️ Got interim results but no final transcript");
        } else {
            println!("\n❌ FAILURE: Got results but all were empty");
        }
    } else {
        println!("\n❌ FAILURE: No transcription results received");
        println!("   Speech → ??? (NO TEXT OUTPUT)");
    }
    
    // Close stream
    println!("\n🔚 Closing stream...");
    let _ = stream.close().await;
    println!("✅ Test completed!");
    
    Ok(())
}