//! Full pipeline example: STT -> Process -> TTS

use debabelizer::{AudioData, AudioFormat, SynthesisOptions, VoiceProcessor};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();
    
    println!("Debabelizer Full Pipeline Example");
    println!("=================================");
    println!("STT -> Text Processing -> TTS\n");
    
    // Create voice processor with custom providers
    let processor = VoiceProcessor::builder()
        .stt_provider("soniox")
        .tts_provider("elevenlabs")
        .build()
        .await?;
    
    // Step 1: Transcribe audio
    println!("Step 1: Transcribing audio...");
    
    let audio_path = "examples/input.wav";
    let transcribed_text = if std::path::Path::new(audio_path).exists() {
        let audio_data = fs::read(audio_path)?;
        let audio = AudioData::new(audio_data, AudioFormat::wav(16000));
        
        match processor.transcribe(audio).await {
            Ok(result) => {
                println!("Transcribed: {}", result.text);
                if let Some(lang) = result.language_detected {
                    println!("Language: {}", lang);
                }
                result.text
            }
            Err(e) => {
                eprintln!("Transcription failed: {}", e);
                "Hello, this is a test of the voice processing pipeline.".to_string()
            }
        }
    } else {
        println!("No input audio found, using default text.");
        "Hello, this is a test of the voice processing pipeline.".to_string()
    };
    
    // Step 2: Process the text (example: convert to uppercase and add emphasis)
    println!("\nStep 2: Processing text...");
    let processed_text = format!(
        "You said: {}. Let me repeat that back to you with emphasis!",
        transcribed_text
    );
    println!("Processed: {}", processed_text);
    
    // Step 3: Synthesize the processed text
    println!("\nStep 3: Synthesizing speech...");
    
    // Get available voices
    let voices = processor.list_voices().await?;
    if voices.is_empty() {
        println!("No voices available for synthesis.");
        return Ok(());
    }
    
    // Select a voice (preferably one that matches the detected language)
    let selected_voice = voices.into_iter().next().unwrap();
    println!("Using voice: {}", selected_voice.name);
    
    let options = SynthesisOptions::new(selected_voice);
    
    match processor.synthesize(&processed_text, &options).await {
        Ok(result) => {
            println!("Synthesis successful!");
            println!("Format: {:?}", result.format);
            println!("Size: {} bytes", result.size_bytes);
            
            // Save the output
            let output_path = "examples/pipeline_output.mp3";
            fs::write(output_path, &result.audio_data)?;
            println!("\nOutput saved to: {}", output_path);
        }
        Err(e) => {
            eprintln!("Synthesis failed: {}", e);
        }
    }
    
    println!("\nPipeline completed successfully!");
    
    Ok(())
}