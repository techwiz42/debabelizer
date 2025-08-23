// Test sending end-of-stream signal to Soniox
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration, sleep};
use uuid::Uuid;

use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use debabelizer_core::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Rust Soniox End-of-Stream Test");
    println!("{}", "=".repeat(50));
    
    // Set up API key
    let api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95";
    
    // Create provider config
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview-v2"));
    let provider_config = ProviderConfig::Simple(config_map);
    
    // Create provider
    let provider = SonioxProvider::new(&provider_config).await?;
    
    // Load test audio
    let audio_data = std::fs::read("/home/peter/debabelizer/english_sample.wav")?;
    
    // Skip WAV header (44 bytes) to get raw PCM data
    let pcm_data = &audio_data[44..];
    println!("📊 Audio: {} PCM bytes", pcm_data.len());
    
    // Create stream config
    let stream_config = StreamConfig {
        session_id: Uuid::new_v4(),
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
    
    let mut stream = provider.transcribe_stream(stream_config).await?;
    println!("✅ Stream created!");
    
    // Send all audio at once
    println!("\n📤 Sending all audio at once ({} bytes)...", pcm_data.len());
    stream.send_audio(pcm_data).await?;
    
    // Try different end-of-stream signals
    println!("\n🔚 Testing end-of-stream signals:");
    
    // 1. Send empty audio chunk
    println!("  1️⃣ Sending empty audio chunk...");
    stream.send_audio(&[]).await?;
    
    // Wait for results
    let mut got_result = false;
    for i in 1..=3 {
        match timeout(Duration::from_secs(2), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                let text = result.text.trim();
                println!("  📥 Got result after empty chunk: '{}' (final: {})", text, result.is_final);
                got_result = true;
                if result.is_final && !text.is_empty() {
                    println!("\n🎉 SUCCESS! Empty chunk triggered final result!");
                    println!("📝 TRANSCRIPT: \"{}\"", text);
                    break;
                }
            }
            _ => {
                if i == 3 && !got_result {
                    println!("  ❌ No result after empty chunk");
                }
            }
        }
    }
    
    // 2. Wait longer for results
    if !got_result {
        println!("\n  2️⃣ Waiting longer (10 seconds)...");
        match timeout(Duration::from_secs(10), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                let text = result.text.trim();
                println!("  📥 Got result after waiting: '{}' (final: {})", text, result.is_final);
                if !text.is_empty() {
                    println!("\n🎉 SUCCESS! Patience triggered result!");
                    println!("📝 TRANSCRIPT: \"{}\"", text);
                    got_result = true;
                }
            }
            _ => {
                println!("  ❌ No result after waiting");
            }
        }
    }
    
    // 3. Close the stream and see if we get results
    if !got_result {
        println!("\n  3️⃣ Closing stream to trigger results...");
        
        // Spawn a task to close the stream after a short delay
        let close_handle = tokio::spawn(async move {
            sleep(Duration::from_millis(500)).await;
            println!("  🔒 Closing stream now...");
            let _ = stream.close().await;
        });
        
        // Try to get results before close
        match timeout(Duration::from_secs(2), stream.receive_transcript()).await {
            Ok(Ok(Some(result))) => {
                let text = result.text.trim();
                println!("  📥 Got result before close: '{}' (final: {})", text, result.is_final);
                if !text.is_empty() {
                    println!("\n🎉 SUCCESS! Pre-close triggered result!");
                    println!("📝 TRANSCRIPT: \"{}\"", text);
                    got_result = true;
                }
            }
            _ => {
                println!("  ❌ No result before close");
            }
        }
        
        close_handle.await?;
    }
    
    if !got_result {
        println!("\n❌ FAILURE: No transcription results with any method");
        println!("   Speech → ??? (NO TEXT OUTPUT)");
    }
    
    println!("\n✅ Test completed!");
    Ok(())
}