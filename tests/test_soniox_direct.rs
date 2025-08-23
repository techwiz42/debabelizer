// Direct Soniox WebSocket streaming test - minimal dependencies
use std::collections::HashMap;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde_json::json;
use url::Url;
use futures::{SinkExt, StreamExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Direct Rust Soniox WebSocket Streaming Test");
    println!("{}", "=".repeat(50));
    
    // Get API key
    let api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95";
    println!("✅ Using Soniox API key: {}***", &api_key[..6]);
    
    // WebSocket URL
    let ws_url = "wss://stt-rt.soniox.com/transcribe-websocket";
    println!("🔗 Connecting to: {}", ws_url);
    
    // Create WebSocket connection with Bearer auth
    let url = Url::parse(ws_url)?;
    let request = tokio_tungstenite::tungstenite::handshake::client::Request::builder()
        .uri(url.as_str())
        .header("Authorization", format!("Bearer {}", api_key))
        .body(())?;
    
    let (mut ws_stream, _) = connect_async(request).await?;
    println!("✅ WebSocket connection established");
    
    // Send configuration
    let config = json!({
        "api_key": api_key,
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "model": "stt-rt-preview-v2",
        "enable_language_identification": true,
        "include_nonfinal": true,
    });
    
    println!("📤 Sending configuration...");
    ws_stream.send(Message::Text(config.to_string())).await?;
    
    // Wait for handshake response
    println!("⏳ Waiting for handshake response...");
    if let Some(msg) = ws_stream.next().await {
        match msg? {
            Message::Text(response) => {
                println!("📥 Handshake response: {}", response);
                
                // Check for errors
                if let Ok(json_response) = serde_json::from_str::<serde_json::Value>(&response) {
                    if let Some(error) = json_response.get("error") {
                        println!("❌ Handshake error: {}", error);
                        return Err(format!("Handshake failed: {}", error).into());
                    }
                }
                println!("✅ Handshake successful");
            }
            other => {
                println!("⚠️ Unexpected handshake message: {:?}", other);
            }
        }
    } else {
        return Err("No handshake response received".into());
    }
    
    // Load test audio
    println!("\n🎵 Loading test audio...");
    let audio_data = std::fs::read("/home/peter/debabelizer/english_sample.wav")
        .or_else(|_| std::fs::read("/home/peter/debabelizer/test.wav"))
        .expect("Failed to load test audio file");
    println!("✅ Loaded {} bytes of audio data", audio_data.len());
    
    // Send audio in chunks and collect responses
    let chunk_size = 3200; // Small chunks to simulate streaming
    let total_chunks = (audio_data.len() + chunk_size - 1) / chunk_size;
    
    println!("\n📤 Sending audio in {} chunks...", total_chunks);
    
    let mut responses = Vec::new();
    let mut chunks_sent = 0;
    
    // Process audio chunks and responses
    for chunk in audio_data.chunks(chunk_size) {
        chunks_sent += 1;
        println!("  📤 Sending chunk {}/{} ({} bytes)", chunks_sent, total_chunks, chunk.len());
        
        // Send audio chunk
        ws_stream.send(Message::Binary(chunk.to_vec())).await?;
        
        // Small delay between chunks
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Check for any responses
        let timeout = tokio::time::Duration::from_millis(500);
        while let Ok(Some(msg)) = tokio::time::timeout(timeout, ws_stream.next()).await {
            match msg? {
                Message::Text(response) => {
                    println!("  📥 Response: {}", response);
                    responses.push(response.clone());
                    
                    // Parse and display transcription
                    if let Ok(json_response) = serde_json::from_str::<serde_json::Value>(&response) {
                        if let Some(tokens) = json_response.get("tokens") {
                            if let Some(tokens_array) = tokens.as_array() {
                                if !tokens_array.is_empty() {
                                    let text: String = tokens_array.iter()
                                        .filter_map(|token| token.get("text")?.as_str())
                                        .collect::<Vec<&str>>()
                                        .join("");
                                    if !text.trim().is_empty() {
                                        println!("    🎯 Transcription: '{}'", text.trim());
                                    }
                                }
                            }
                        }
                        
                        if let Some(error) = json_response.get("error") {
                            println!("    ❌ Error in response: {}", error);
                        }
                    }
                }
                Message::Close(_) => {
                    println!("  🔚 WebSocket closed by server");
                    break;
                }
                other => {
                    println!("  📨 Other message: {:?}", other);
                }
            }
        }
    }
    
    // Wait for final responses
    println!("\n⏳ Waiting for final responses...");
    let final_timeout = tokio::time::Duration::from_secs(3);
    let deadline = tokio::time::Instant::now() + final_timeout;
    
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(msg)) = tokio::time::timeout(tokio::time::Duration::from_millis(200), ws_stream.next()).await {
            match msg? {
                Message::Text(response) => {
                    println!("  📥 Final response: {}", response);
                    responses.push(response);
                }
                Message::Close(_) => {
                    println!("  🔚 WebSocket closed");
                    break;
                }
                _ => {}
            }
        }
    }
    
    // Close connection
    println!("\n🔚 Closing WebSocket connection...");
    ws_stream.close(None).await?;
    
    // Analyze results
    println!("\n📊 Results Analysis:");
    println!("  📋 Total responses received: {}", responses.len());
    
    let mut final_transcript = String::new();
    let mut total_tokens = 0;
    
    for (i, response) in responses.iter().enumerate() {
        if let Ok(json_response) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(tokens) = json_response.get("tokens") {
                if let Some(tokens_array) = tokens.as_array() {
                    total_tokens += tokens_array.len();
                    
                    for token in tokens_array {
                        if let (Some(text), Some(is_final)) = (
                            token.get("text").and_then(|t| t.as_str()),
                            token.get("is_final").and_then(|f| f.as_bool())
                        ) {
                            if is_final {
                                final_transcript.push_str(text);
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("  🎯 Total tokens processed: {}", total_tokens);
    println!("  📝 Final transcript: '{}'", final_transcript.trim());
    println!("  📏 Transcript length: {} characters", final_transcript.trim().len());
    
    if final_transcript.trim().is_empty() {
        println!("  ⚠️ No final transcription results - this may indicate:");
        println!("    - Audio format incompatibility");
        println!("    - Insufficient audio content");
        println!("    - API configuration issues");
    } else {
        println!("  ✅ Transcription successful!");
    }
    
    println!("\n🎉 Direct Soniox streaming test completed!");
    println!("{}", "=".repeat(50));
    
    Ok(())
}