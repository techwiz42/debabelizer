// Low-level WebSocket debugging for Soniox
use std::collections::HashMap;
use serde_json::json;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Direct WebSocket Test with Soniox");
    println!("{}", "=".repeat(50));
    
    let api_key = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95";
    let url = format!("wss://stt-rt.soniox.com/transcribe-websocket?api_key={}", api_key);
    
    println!("📡 Connecting to Soniox WebSocket...");
    let (mut ws_stream, _) = connect_async(&url).await?;
    println!("✅ Connected!");
    
    // Send configuration - NO MODEL AT ALL
    let config = json!({
        "api_key": api_key,
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "include_nonfinal": true,
    });
    
    println!("\n📤 Sending configuration...");
    ws_stream.send(Message::Text(config.to_string())).await?;
    
    // Wait for handshake
    if let Some(Ok(Message::Text(response))) = ws_stream.next().await {
        println!("📥 Handshake response: {}", response);
    }
    
    // Load audio
    let audio_data = std::fs::read("/home/peter/debabelizer/english_sample.wav")?;
    let pcm_data = &audio_data[44..]; // Skip WAV header
    
    println!("\n📊 Loaded {} bytes of PCM audio", pcm_data.len());
    
    // Send audio in small chunks
    let chunk_size = 8000; // 0.5 seconds at 16kHz
    let chunks: Vec<_> = pcm_data.chunks(chunk_size).collect();
    
    // Start receiving task
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let receive_task = tokio::spawn(async move {
        let mut count = 0;
        while let Some(msg) = rx.recv().await {
            count += 1;
            println!("  📥 Message #{}: {}", count, msg);
        }
        println!("  🔚 Receive task ended after {} messages", count);
    });
    
    // Send chunks and check for responses
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n📤 Sending chunk {}/{} ({} bytes)", i + 1, chunks.len(), chunk.len());
        ws_stream.send(Message::Binary(chunk.to_vec())).await?;
        
        // Check for any immediate responses
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Try to receive with timeout
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(500),
            ws_stream.next()
        ).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                tx.send(format!("After chunk {}: {}", i + 1, text))?;
            }
            Ok(Some(Ok(msg))) => {
                tx.send(format!("After chunk {}: {:?}", i + 1, msg))?;
            }
            Err(_) => {
                // Timeout - no message
            }
            _ => {}
        }
    }
    
    // Try sending empty chunk as end-of-stream
    println!("\n📤 Sending empty chunk as end-of-stream signal...");
    ws_stream.send(Message::Binary(vec![])).await?;
    
    // Send a few more empty chunks to signal end of audio
    println!("\n📤 Sending multiple empty chunks to signal end of stream...");
    for i in 0..3 {
        ws_stream.send(Message::Binary(vec![])).await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    // Wait for more responses
    println!("\n⏳ Waiting for final responses (10 seconds)...");
    let start = tokio::time::Instant::now();
    while start.elapsed() < tokio::time::Duration::from_secs(10) {
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(500),
            ws_stream.next()
        ).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                tx.send(format!("Final wait: {}", text))?;
            }
            Ok(Some(Ok(Message::Close(_)))) => {
                tx.send("WebSocket closed by server".to_string())?;
                break;
            }
            Ok(Some(Ok(msg))) => {
                tx.send(format!("Final wait: {:?}", msg))?;
            }
            _ => {}
        }
    }
    
    // Close WebSocket
    println!("\n🔚 Closing WebSocket...");
    ws_stream.close(None).await?;
    
    drop(tx);
    receive_task.await?;
    
    println!("\n✅ Test completed!");
    Ok(())
}