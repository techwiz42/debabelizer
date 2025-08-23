use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SONIOX_API_KEY")
        .expect("SONIOX_API_KEY environment variable not set");
    
    println!("🌐 Testing API key in message (Python approach)...");
    
    // Connect directly without any headers or query parameters
    match connect_async("wss://stt-rt.soniox.com/transcribe-websocket").await {
        Ok((mut ws_stream, response)) => {
            println!("✅ Connected! Response: {:?}", response.status());
            
            // Send initial configuration with API key in message (like Python)
            let init_message = json!({
                "api_key": api_key,  // API key in the message
                "audio_format": "pcm_s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "model": "stt-rt-preview-v2",
                "enable_language_identification": true,
                "include_nonfinal": true
            });
            
            println!("📤 Sending config with API key in message: {}", init_message);
            
            if let Err(e) = ws_stream.send(Message::Text(init_message.to_string())).await {
                println!("❌ Failed to send config: {}", e);
                return Ok(());
            }
            
            // Wait for response
            println!("⏳ Waiting for handshake response...");
            
            match ws_stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    println!("✅ Received handshake: {}", text);
                    
                    // Check for success
                    if text.contains("\"error_code\":401") || text.contains("Missing API key") {
                        println!("❌ Still getting authentication error!");
                    } else {
                        println!("🎉 Authentication SUCCESS!");
                    }
                }
                Some(Ok(msg)) => {
                    println!("🔍 Received non-text message: {:?}", msg);
                }
                Some(Err(e)) => {
                    println!("❌ Error receiving message: {}", e);
                }
                None => {
                    println!("❌ Connection closed");
                }
            }
            
            let _ = ws_stream.close(None).await;
        }
        Err(e) => {
            println!("❌ Connection failed: {}", e);
        }
    }
    
    Ok(())
}