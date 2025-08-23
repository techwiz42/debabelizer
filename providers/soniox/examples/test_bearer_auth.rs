use tokio_tungstenite::{connect_async, tungstenite::{Message, handshake::client::Request}};
use futures::{SinkExt, StreamExt};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("SONIOX_API_KEY")
        .expect("SONIOX_API_KEY environment variable not set");
    
    println!("🌐 Testing Bearer authentication...");
    
    // Use Bearer token authentication in WebSocket headers
    let request = Request::builder()
        .uri("wss://stt-rt.soniox.com/transcribe-websocket")
        .header("Authorization", format!("Bearer {}", api_key))
        .body(())?;
    
    match connect_async(request).await {
        Ok((mut ws_stream, response)) => {
            println!("✅ Connected with Bearer auth! Response: {:?}", response.status());
            
            // Send initial configuration
            let init_message = json!({
                "audio_format": "pcm_s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "model": "stt-rt-preview-v2",
                "enable_language_identification": true,
                "include_nonfinal": true
            });
            
            println!("📤 Sending config: {}", init_message);
            
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