//! Basic transcription example using Debabelizer

use debabelizer::{AudioData, AudioFormat, VoiceProcessor};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();
    
    println!("Debabelizer Basic Transcription Example");
    println!("======================================\n");
    
    // Create a voice processor with default configuration
    let processor = VoiceProcessor::new()?;
    
    // List available STT providers
    let stt_providers = processor.list_stt_providers().await?;
    println!("Available STT providers: {:?}\n", stt_providers);
    
    // Read audio file
    let audio_path = "examples/sample.wav";
    println!("Reading audio file: {}", audio_path);
    
    let audio_data = if std::path::Path::new(audio_path).exists() {
        fs::read(audio_path)?
    } else {
        println!("Sample audio file not found. Using dummy data for demonstration.");
        vec![0u8; 1000] // Dummy data
    };
    
    // Create AudioData with WAV format
    let audio = AudioData::new(audio_data, AudioFormat::wav(16000));
    
    // Transcribe the audio
    println!("Transcribing audio...");
    match processor.transcribe(audio).await {
        Ok(result) => {
            println!("\nTranscription Result:");
            println!("Text: {}", result.text);
            println!("Confidence: {:.2}%", result.confidence * 100.0);
            
            if let Some(language) = result.language_detected {
                println!("Language detected: {}", language);
            }
            
            if let Some(duration) = result.duration {
                println!("Duration: {:.2} seconds", duration);
            }
            
            if let Some(words) = result.words {
                println!("\nWord timings:");
                for word in words.iter().take(10) {
                    println!("  {}: {:.2}s - {:.2}s (confidence: {:.2}%)",
                        word.word, word.start, word.end, word.confidence * 100.0);
                }
                if words.len() > 10 {
                    println!("  ... and {} more words", words.len() - 10);
                }
            }
        }
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
        }
    }
    
    Ok(())
}