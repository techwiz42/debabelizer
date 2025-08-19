//! Voice synthesis example using Debabelizer

use debabelizer::{AudioFormat, SynthesisOptions, VoiceProcessor};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();
    
    println!("Debabelizer Voice Synthesis Example");
    println!("===================================\n");
    
    // Create a voice processor with default configuration
    let processor = VoiceProcessor::new()?;
    
    // List available TTS providers
    let tts_providers = processor.list_tts_providers().await?;
    println!("Available TTS providers: {:?}\n", tts_providers);
    
    // List available voices
    println!("Fetching available voices...");
    let voices = processor.list_voices().await?;
    
    if voices.is_empty() {
        println!("No voices available. Please check your provider configuration.");
        return Ok(());
    }
    
    println!("Available voices:");
    for (i, voice) in voices.iter().enumerate().take(5) {
        println!("  {}. {} ({}) - {}",
            i + 1,
            voice.name,
            voice.voice_id,
            voice.language
        );
        if let Some(desc) = &voice.description {
            println!("     {}", desc);
        }
    }
    if voices.len() > 5 {
        println!("  ... and {} more voices", voices.len() - 5);
    }
    
    // Select first voice
    let selected_voice = &voices[0];
    println!("\nUsing voice: {} ({})", selected_voice.name, selected_voice.voice_id);
    
    // Text to synthesize
    let text = "Hello! This is a test of the Debabelizer voice synthesis system. \
                It supports multiple providers and voices for text-to-speech conversion.";
    
    println!("\nText to synthesize:");
    println!("{}", text);
    
    // Create synthesis options
    let options = SynthesisOptions::new(selected_voice.clone());
    
    // Synthesize speech
    println!("\nSynthesizing speech...");
    match processor.synthesize(text, &options).await {
        Ok(result) => {
            println!("\nSynthesis Result:");
            println!("Format: {:?}", result.format);
            println!("Size: {} bytes", result.size_bytes);
            
            if let Some(duration) = result.duration {
                println!("Duration: {:.2} seconds", duration);
            }
            
            // Save audio to file
            let output_path = "examples/output.mp3";
            fs::write(output_path, &result.audio_data)?;
            println!("\nAudio saved to: {}", output_path);
        }
        Err(e) => {
            eprintln!("Synthesis failed: {}", e);
        }
    }
    
    Ok(())
}