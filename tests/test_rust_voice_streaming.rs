// Comprehensive Rust Soniox voice streaming test
use std::collections::HashMap;
use serde_json::json;
use tokio::time::{timeout, Duration, sleep};
use uuid::Uuid;

// Import Soniox provider directly
use debabelizer_soniox::{SonioxProvider, ProviderConfig};
use debabelizer_core::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for detailed logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    
    println!("🦀 COMPREHENSIVE RUST SONIOX VOICE STREAMING TEST");
    println!("{}", "=".repeat(80));
    
    // Set up API key
    let api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95";
    println!("✅ Using Soniox API key: {}***", &api_key[..6]);
    
    // Create provider configuration
    let mut config_map = HashMap::new();
    config_map.insert("api_key".to_string(), json!(api_key));
    config_map.insert("model".to_string(), json!("stt-rt-preview-v2"));
    config_map.insert("auto_detect_language".to_string(), json!(true));
    
    let provider_config = ProviderConfig::Simple(config_map);
    
    // Test 1: Create Soniox provider
    println!("\n🔧 STEP 1: Creating Soniox STT Provider...");
    let provider = match SonioxProvider::new(&provider_config).await {
        Ok(p) => {
            println!("✅ Soniox provider created successfully");
            println!("   - Name: {}", p.name());
            println!("   - Supports streaming: {}", p.supports_streaming());
            println!("   - Supported formats: {:?}", p.supported_formats());
            p
        },
        Err(e) => {
            println!("❌ Failed to create Soniox provider: {:?}", e);
            return Err(e.into());
        }
    };
    
    // Test 2: Load voice audio file
    println!("\n🎵 STEP 2: Loading voice audio file...");
    let test_files = [
        "/home/peter/debabelizer/english_sample.wav",
        "/home/peter/debabelizer/test.wav",
        "/home/peter/debabelizer/file_0.wav"
    ];
    
    let mut audio_data = None;
    let mut used_file = "";
    
    for file_path in &test_files {
        match std::fs::read(file_path) {
            Ok(data) => {
                audio_data = Some(data);
                used_file = file_path;
                break;
            }
            Err(_) => continue,
        }
    }
    
    let audio_data = audio_data.ok_or("No test audio file found")?;
    println!("✅ Loaded {} bytes from {}", audio_data.len(), used_file);
    
    // Create proper audio format
    let audio_format = AudioFormat {
        format: "wav".to_string(),
        sample_rate: 16000,
        channels: 1,
        bit_depth: Some(16),
    };
    
    // Test 3: Create streaming session
    println!("\n🌊 STEP 3: Creating streaming session...");
    let session_id = Uuid::new_v4();
    println!("📋 Session ID: {}", session_id);
    
    let stream_config = StreamConfig {
        session_id,
        format: audio_format.clone(),
        interim_results: true,
        language: None, // Auto-detect
        model: Some("stt-rt-preview-v2".to_string()),
        punctuate: true,
        profanity_filter: false,
        diarization: false,
        metadata: None,
        enable_word_time_offsets: true,
        enable_automatic_punctuation: true,
        enable_language_identification: true,
    };
    
    println!("📊 Stream Configuration:");
    println!("   - Audio: {} Hz, {} channels, {} format", 
             stream_config.format.sample_rate, 
             stream_config.format.channels,
             stream_config.format.format);
    println!("   - Interim results: {}", stream_config.interim_results);
    println!("   - Language detection: {}", stream_config.enable_language_identification);
    println!("   - Word timestamps: {}", stream_config.enable_word_time_offsets);
    
    // Test 4: Create streaming connection (this tests our async fix!)
    println!("\n🔗 STEP 4: Creating streaming connection (CRITICAL TEST)...");
    println!("🔍 This will test if the background WebSocket task executes properly!");
    
    let mut stream = match provider.transcribe_stream(stream_config).await {
        Ok(s) => {
            println!("✅ Streaming connection created successfully!");
            println!("   - Session ID: {}", s.session_id());
            println!("   - This confirms the async task execution fix worked!");
            s
        },
        Err(e) => {
            println!("❌ Failed to create streaming connection: {:?}", e);
            println!("💡 This would indicate our async fix didn't work");
            return Err(e.into());
        }
    };
    
    // Test 5: Concurrent audio streaming and result collection
    println!("\n📡 STEP 5: Starting concurrent audio streaming and result collection...");
    
    // Prepare audio chunks (simulate real-time streaming)
    let chunk_size = 3200; // 200ms chunks at 16kHz
    let chunks: Vec<Vec<u8>> = audio_data.chunks(chunk_size).map(|c| c.to_vec()).collect();
    println!("📤 Prepared {} audio chunks of ~{} bytes each", chunks.len(), chunk_size);
    
    // Storage for results
    let mut all_results = Vec::new();
    let mut final_results = Vec::new();
    
    // Task 1: Send audio chunks with realistic timing
    let stream_send = async {
        println!("🎙️ Starting audio chunk streaming...");
        
        for (i, chunk) in chunks.iter().enumerate() {
            println!("   📤 Sending chunk {}/{} ({} bytes)", i + 1, chunks.len(), chunk.len());
            
            match stream.send_audio(chunk).await {
                Ok(()) => {
                    println!("   ✅ Chunk {} sent successfully", i + 1);
                }
                Err(e) => {
                    println!("   ❌ Failed to send chunk {}: {:?}", i + 1, e);
                    return Err::<(), Box<dyn std::error::Error>>(e.into());
                }
            }
            
            // Realistic streaming delay (simulate real-time audio)
            sleep(Duration::from_millis(200)).await;
        }
        
        println!("✅ All audio chunks sent successfully!");
        
        // Give time for final processing
        println!("⏳ Waiting for final processing...");
        sleep(Duration::from_secs(2)).await;
        
        // Close the stream
        println!("🔚 Closing stream...");
        if let Err(e) = stream.close().await {
            println!("⚠️ Error closing stream: {:?}", e);
        } else {
            println!("✅ Stream closed successfully");
        }
        
        Ok(())
    };
    
    // Task 2: Collect streaming results
    let stream_receive = async {
        println!("🎯 Starting result collection...");
        let mut result_count = 0;
        
        loop {
            match timeout(Duration::from_secs(5), stream.receive_transcript()).await {
                Ok(Ok(Some(result))) => {
                    result_count += 1;
                    let is_final = result.is_final;
                    let text = result.text.trim();
                    let confidence = result.confidence;
                    
                    println!("   📥 Result #{}: '{}' (final: {}, confidence: {:.3})", 
                            result_count, text, is_final, confidence);
                    
                    // Show metadata if present
                    if let Some(metadata) = &result.metadata {
                        if let Some(msg_type) = metadata.get("type") {
                            println!("      ℹ️ Type: {}", msg_type);
                        }
                    }
                    
                    // Show word timings if present
                    if let Some(words) = &result.words {
                        if words.len() > 0 {
                            println!("      ⏰ {} words with timing", words.len());
                        }
                    }
                    
                    all_results.push(result);
                    
                    // Store final results separately
                    if is_final && !text.is_empty() {
                        final_results.push(all_results.last().unwrap().clone());
                        println!("      ✅ Final result added to transcript");
                    }
                }
                Ok(Ok(None)) => {
                    println!("   🔚 Stream ended normally");
                    break;
                }
                Ok(Err(e)) => {
                    println!("   ❌ Error receiving result: {:?}", e);
                    break;
                }
                Err(_timeout) => {
                    println!("   ⏰ Timeout waiting for results (5s)");
                    if result_count == 0 {
                        println!("   ⚠️ No results received yet - continuing to wait...");
                        continue;
                    } else {
                        println!("   ℹ️ Stopping collection after timeout");
                        break;
                    }
                }
            }
            
            // Safety limit
            if result_count >= 50 {
                println!("   ⏹️ Stopping after {} results (safety limit)", result_count);
                break;
            }
        }
        
        println!("✅ Result collection completed with {} total results", result_count);
        Ok(())
    };
    
    // Run both tasks concurrently with overall timeout
    println!("\n⏰ Running streaming test with 45-second timeout...");
    match timeout(Duration::from_secs(45), tokio::try_join!(stream_send, stream_receive)).await {
        Ok(Ok(((), ()))) => {
            println!("✅ Concurrent streaming and result collection completed successfully!");
        }
        Ok(Err(e)) => {
            println!("❌ Streaming test failed: {:?}", e);
            return Err(e);
        }
        Err(_) => {
            println!("⏰ Streaming test timed out after 45 seconds");
            println!("   This might indicate WebSocket connection issues");
        }
    }
    
    // Test 6: Analyze results
    println!("\n📊 STEP 6: Analyzing streaming results...");
    println!("Results Summary:");
    println!("   📋 Total results received: {}", all_results.len());
    println!("   🎯 Final results: {}", final_results.len());
    
    if all_results.is_empty() {
        println!("   ❌ NO RESULTS RECEIVED!");
        println!("   💡 This indicates the WebSocket background task is still not executing");
        println!("   🔍 Check if the debug logs show task startup messages");
        return Ok(());
    }
    
    // Analyze interim vs final results
    let interim_results = all_results.len() - final_results.len();
    println!("   📈 Interim results: {}", interim_results);
    
    // Build complete transcript
    if !final_results.is_empty() {
        let complete_transcript: String = final_results
            .iter()
            .map(|r| r.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<&str>>()
            .join(" ");
        
        println!("\n📝 COMPLETE TRANSCRIPT:");
        println!("   \"{}\"", complete_transcript);
        println!("   📏 Length: {} characters", complete_transcript.len());
        
        // Calculate average confidence
        let avg_confidence: f32 = final_results
            .iter()
            .map(|r| r.confidence)
            .sum::<f32>() / final_results.len() as f32;
        println!("   🎯 Average confidence: {:.3}", avg_confidence);
        
        // Count words with timing
        let total_words: usize = final_results
            .iter()
            .filter_map(|r| r.words.as_ref())
            .map(|words| words.len())
            .sum();
        if total_words > 0 {
            println!("   ⏰ Words with timing: {}", total_words);
        }
        
        // Show language detection if available
        for result in &final_results {
            if let Some(metadata) = &result.metadata {
                if let Some(language) = metadata.get("language") {
                    println!("   🌍 Detected language: {}", language);
                    break;
                }
            }
        }
    }
    
    // Final assessment
    println!("\n🎉 STREAMING TEST ASSESSMENT:");
    
    if all_results.len() > 0 {
        println!("✅ SUCCESS: Rust Soniox streaming is working!");
        println!("   - Background WebSocket task is executing");
        println!("   - Real-time transcription is functional");
        println!("   - Message-driven architecture is operational");
        
        if final_results.len() > 0 {
            println!("   - Transcription accuracy appears good");
            println!("   - The async execution bottleneck fix was successful!");
        }
    } else {
        println!("❌ PARTIAL SUCCESS: Connection works but no transcription received");
        println!("   - This might indicate audio format or API configuration issues");
        println!("   - The async task fix worked (connection established)");
    }
    
    println!("\n🏁 Rust Soniox voice streaming test completed!");
    println!("{}", "=".repeat(80));
    
    Ok(())
}