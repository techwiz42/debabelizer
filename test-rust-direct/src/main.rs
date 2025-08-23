use std::collections::HashMap;
use std::env;

use debabelizer::{DebabelizerConfig, VoiceProcessor};
use debabelizer_core::{AudioData, AudioFormat, StreamConfig};
use serde_json::json;
use uuid::Uuid;

fn generate_test_audio() -> Vec<u8> {
    // Generate 2 seconds of simple sine wave audio at 16kHz
    let sample_rate = 16000;
    let duration = 2.0; // seconds
    let frequency = 440.0; // A4 note
    
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut audio_data = Vec::with_capacity(num_samples * 2); // 16-bit samples
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin() * 0.5;
        let sample_i16 = (sample * 32767.0) as i16;
        
        // Convert to little-endian bytes
        audio_data.extend_from_slice(&sample_i16.to_le_bytes());
    }
    
    audio_data
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Testing Rust Debabelizer Implementation Directly\n");
    
    // Set up configuration
    let api_key = env::var("SONIOX_API_KEY")
        .unwrap_or_else(|_| {
            println!("⚠️  SONIOX_API_KEY not set, using dummy key for connection test");
            "dummy-key-for-testing".to_string()
        });
    
    let mut config_map = HashMap::new();
    config_map.insert(
        "soniox".to_string(),
        json!({
            "api_key": api_key,
            "model": "stt-rt-preview",
            "auto_detect_language": true
        })
    );
    
    config_map.insert(
        "preferences".to_string(),
        json!({
            "stt_provider": "soniox",
            "auto_select": false
        })
    );
    
    // Create configuration
    let config = DebabelizerConfig::from_map(config_map)?;
    println!("✅ Created Rust configuration");
    
    // Create processor  
    let mut processor = VoiceProcessor::new(config).await?;
    println!("✅ Created Rust VoiceProcessor");
    
    // Generate test audio
    let audio_data = generate_test_audio();
    println!("🎵 Generated {} bytes of test audio", audio_data.len());
    
    let audio_format = AudioFormat {
        format: "pcm".to_string(),
        sample_rate: 16000,
        channels: 1,
        bit_depth: Some(16),
    };
    
    let audio = AudioData {
        data: audio_data.clone(),
        format: audio_format.clone(),
    };
    
    // Test 1: Batch Transcription
    println!("\n📝 Testing Batch Transcription:");
    match processor.transcribe(audio).await {
        Ok(result) => {
            println!("✅ Transcription successful:");
            println!("   Text: '{}'", result.text);
            println!("   Confidence: {:.2}", result.confidence);
            if let Some(lang) = result.language_detected {
                println!("   Language: {}", lang);
            }
        }
        Err(e) => {
            println!("❌ Batch transcription failed: {}", e);
        }
    }
    
    // Test 2: Streaming Transcription
    println!("\n🎙️  Testing Streaming Transcription:");
    let stream_config = StreamConfig {
        session_id: Uuid::new_v4(),
        format: audio_format,
        interim_results: true,
        language: None,
        model: Some("stt-rt-preview".to_string()),
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: false,
        enable_automatic_punctuation: true,
        enable_language_identification: true,
    };
    
    match processor.transcribe_stream(stream_config).await {
        Ok(mut stream) => {
            println!("✅ Created streaming session: {}", stream.session_id());
            
            // Send audio in chunks
            let chunk_size = 3200; // 0.2 seconds at 16kHz
            let mut chunk_count = 0;
            
            for chunk in audio_data.chunks(chunk_size) {
                chunk_count += 1;
                println!("📤 Sending chunk {} ({} bytes)", chunk_count, chunk.len());
                
                if let Err(e) = stream.send_audio(chunk).await {
                    println!("❌ Failed to send audio chunk {}: {}", chunk_count, e);
                    break;
                }
                
                // Try to receive results
                let mut result_count = 0;
                while result_count < 3 { // Limit attempts to avoid hanging
                    match stream.receive_transcript().await {
                        Ok(Some(result)) => {
                            result_count += 1;
                            println!("   📝 Result {}: text='{}', final={}, confidence={:.2}", 
                                   result_count, result.text, result.is_final, result.confidence);
                            
                            if result.is_final && !result.text.trim().is_empty() {
                                break;
                            }
                        }
                        Ok(None) => {
                            println!("   📝 Stream ended");
                            break;
                        }
                        Err(e) => {
                            println!("   ❌ Error receiving result: {}", e);
                            break;
                        }
                    }
                }
                
                // Small delay to simulate real-time
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
            
            // Close the stream
            if let Err(e) = stream.close().await {
                println!("⚠️  Error closing stream: {}", e);
            } else {
                println!("✅ Stream closed successfully");
            }
        }
        Err(e) => {
            println!("❌ Failed to create streaming session: {}", e);
            
            // Print more detailed error information
            println!("Error details: {:?}", e);
            
            // Check if it's a specific type of error
            match &e {
                debabelizer::DebabelizerError::Configuration(msg) => {
                    println!("Configuration error: {}", msg);
                }
                debabelizer::DebabelizerError::Provider(provider_err) => {
                    println!("Provider error: {:?}", provider_err);
                }
                _ => {
                    println!("Other error type");
                }
            }
        }
    }
    
    println!("\n🦀 Rust direct test completed!");
    Ok(())
}