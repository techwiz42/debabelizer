//! OpenAI TTS Provider Example
//! 
//! This example demonstrates how to use the OpenAI TTS provider with the Debabelizer library.
//! 
//! Usage:
//! ```bash
//! # Set your OpenAI API key
//! export OPENAI_API_KEY="your-api-key-here"
//! 
//! # Run the example
//! cargo run --example openai_tts_example --features openai
//! ```

use debabelizer::{DebabelizerConfig, VoiceProcessor};
use debabelizer_core::{SynthesisOptions, Voice, AudioFormat};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Check for API key
    let api_key = env::var("OPENAI_API_KEY")
        .expect("Please set OPENAI_API_KEY environment variable");
    
    println!("🎤 OpenAI TTS Provider Example");
    println!("===============================");
    
    // Create configuration with OpenAI settings
    let mut config = DebabelizerConfig::default();
    
    // Add OpenAI configuration programmatically
    let mut openai_config = std::collections::HashMap::new();
    openai_config.insert("api_key".to_string(), serde_json::json!(api_key));
    openai_config.insert("tts_model".to_string(), serde_json::json!("tts-1"));
    openai_config.insert("tts_voice".to_string(), serde_json::json!("alloy"));
    
    config.providers.insert(
        "openai".to_string(), 
        debabelizer::config::ProviderConfig::Simple(openai_config)
    );
    
    // Create voice processor
    let processor = VoiceProcessor::with_config(config)?;
    
    // Set OpenAI as the TTS provider
    processor.set_tts_provider("openai").await?;
    
    println!("✅ OpenAI TTS provider initialized successfully");
    
    // List available voices
    println!("\n🎭 Available OpenAI voices:");
    match processor.list_tts_voices().await {
        Ok(voices) => {
            for voice in voices {
                println!("  • {} ({}): {}", 
                    voice.name, 
                    voice.voice_id,
                    voice.description.unwrap_or_else(|| "No description".to_string())
                );
            }
        }
        Err(e) => println!("  ❌ Error listing voices: {}", e),
    }
    
    // Example text to synthesize
    let text = "Hello from OpenAI Text-to-Speech! This is a demonstration of the Debabelizer universal voice processing library.";
    println!("\n📝 Text to synthesize: \"{}\"", text);
    
    // Create synthesis options with different voices
    let voices_to_test = vec!["alloy", "echo", "nova"];
    
    for voice_id in voices_to_test {
        println!("\n🔊 Synthesizing with voice: {}", voice_id);
        
        let voice = Voice::new(voice_id.to_string(), voice_id.to_string(), "en".to_string());
        let mut options = SynthesisOptions::new(voice);
        options.model = Some("tts-1".to_string());
        options.speed = Some(1.0);
        options.format = AudioFormat::mp3(24000);
        
        match processor.synthesize_text(text, &options).await {
            Ok(result) => {
                println!("  ✅ Synthesis successful!");
                println!("     Format: {}", result.format.format);
                println!("     Sample rate: {} Hz", result.format.sample_rate);
                println!("     Duration: {:.2}s", result.duration.unwrap_or(0.0));
                println!("     Size: {} bytes", result.size_bytes);
                
                // Save to file for testing (optional)
                let filename = format!("openai_output_{}.{}", voice_id, result.format.format);
                if let Err(e) = std::fs::write(&filename, &result.audio_data) {
                    println!("     ⚠️  Could not save to {}: {}", filename, e);
                } else {
                    println!("     💾 Saved to: {}", filename);
                }
            }
            Err(e) => {
                println!("  ❌ Synthesis failed: {}", e);
                if e.to_string().contains("401") || e.to_string().contains("unauthorized") {
                    println!("     💡 Tip: Check your OPENAI_API_KEY environment variable");
                } else if e.to_string().contains("429") || e.to_string().contains("rate") {
                    println!("     💡 Tip: You may have hit rate limits. Wait a moment and try again.");
                } else if e.to_string().contains("quota") || e.to_string().contains("billing") {
                    println!("     💡 Tip: Check your OpenAI account billing and quotas");
                }
                break;
            }
        }
        
        // Small delay between requests to be respectful to the API
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    
    // Demonstrate streaming synthesis
    println!("\n🌊 Testing streaming synthesis...");
    let voice = Voice::new("shimmer".to_string(), "Shimmer".to_string(), "en".to_string());
    let options = SynthesisOptions::new(voice);
    
    match processor.synthesize_text_stream(text, &options).await {
        Ok(mut stream) => {
            println!("  ✅ Streaming started");
            let mut chunk_count = 0;
            let mut total_bytes = 0;
            
            while let Ok(Some(chunk)) = stream.receive_chunk().await {
                chunk_count += 1;
                total_bytes += chunk.len();
                println!("    📦 Received chunk {}: {} bytes", chunk_count, chunk.len());
            }
            
            println!("  ✅ Streaming complete: {} chunks, {} total bytes", chunk_count, total_bytes);
            let _ = stream.close().await;
        }
        Err(e) => {
            println!("  ❌ Streaming failed: {}", e);
        }
    }
    
    println!("\n🎉 OpenAI TTS example completed!");
    println!("\n💡 Tips:");
    println!("   • OpenAI TTS supports text up to 4096 characters");
    println!("   • Use 'tts-1' for speed or 'tts-1-hd' for quality");
    println!("   • Supported formats: mp3, opus, aac, flac, wav, pcm");
    println!("   • All voices are optimized for English but work with other languages");
    
    Ok(())
}