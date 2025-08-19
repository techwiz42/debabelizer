//! Streaming STT example using Debabelizer

use debabelizer::{AudioFormat, StreamConfig, VoiceProcessor};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();
    
    println!("Debabelizer Streaming STT Example");
    println!("=================================\n");
    
    // Create a voice processor
    let processor = VoiceProcessor::new()?;
    
    // Configure streaming
    let config = StreamConfig {
        format: AudioFormat::wav(16000),
        interim_results: true,
        punctuate: true,
        language: None, // Auto-detect
        ..Default::default()
    };
    
    println!("Starting streaming transcription...");
    println!("Session ID: {}", config.session_id);
    println!("Format: {:?}", config.format);
    println!("Interim results: {}", config.interim_results);
    println!();
    
    // Create streaming session
    let mut stream = processor.transcribe_stream(config).await?;
    
    // Simulate sending audio chunks
    println!("Simulating audio stream...");
    
    // In a real application, you would read from microphone or audio stream
    // Here we'll just send some dummy data for demonstration
    for i in 0..5 {
        println!("Sending audio chunk {}...", i + 1);
        
        // Create dummy audio chunk (in real app, this would be actual audio data)
        let chunk = vec![0u8; 3200]; // 100ms of 16kHz mono audio
        
        stream.send_audio(&chunk).await?;
        
        // Check for transcripts
        while let Ok(Some(result)) = stream.receive_transcript().await {
            if result.is_final {
                println!("[FINAL] {}", result.text);
                if let Some(words) = &result.words {
                    println!("  Words: {} total", words.len());
                }
            } else {
                println!("[INTERIM] {}", result.text);
            }
            println!("  Confidence: {:.2}%", result.confidence * 100.0);
            
            if let Some(metadata) = &result.metadata {
                if let Some(language) = metadata.get("language").and_then(|v| v.as_str()) {
                    println!("  Language: {}", language);
                }
            }
        }
        
        // Simulate delay between chunks
        sleep(Duration::from_millis(100)).await;
    }
    
    // Close the stream
    println!("\nClosing stream...");
    stream.close().await?;
    
    println!("Streaming session completed.");
    
    Ok(())
}