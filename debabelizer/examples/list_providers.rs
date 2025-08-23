use debabelizer::{DebabelizerConfig, VoiceProcessor};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up some dummy environment variables for testing
    env::set_var("SONIOX_API_KEY", "test_key");
    env::set_var("ELEVENLABS_API_KEY", "test_key");
    env::set_var("DEEPGRAM_API_KEY", "test_key");
    env::set_var("OPENAI_API_KEY", "test_key");
    env::set_var("GOOGLE_API_KEY", "test_key");
    env::set_var("AZURE_SPEECH_KEY", "test_key");
    env::set_var("AZURE_SPEECH_REGION", "eastus");
    
    println!("Environment variables set:");
    println!("SONIOX_API_KEY: {}", env::var("SONIOX_API_KEY").unwrap_or_default());
    println!("ELEVENLABS_API_KEY: {}", env::var("ELEVENLABS_API_KEY").unwrap_or_default());
    println!("DEEPGRAM_API_KEY: {}", env::var("DEEPGRAM_API_KEY").unwrap_or_default());
    
    println!("\nCreating VoiceProcessor with default config...");
    
    let config = DebabelizerConfig::new()?;
    
    // Debug: Check what's in the config
    println!("\nChecking config providers:");
    for provider in ["soniox", "elevenlabs", "deepgram", "openai", "google", "azure", "whisper"] {
        if let Some(prov_config) = config.get_provider_config(provider) {
            println!("  {} config found: {:?}", provider, prov_config);
        } else {
            println!("  {} config NOT found", provider);
        }
    }
    
    let processor = VoiceProcessor::with_config(config)?;
    
    println!("\nVoiceProcessor created successfully!");
    
    // Try to list available providers
    println!("\nTrying to list available providers...");
    let available_stt = processor.list_stt_providers().await?;
    let available_tts = processor.list_tts_providers().await?;
    
    println!("Available STT providers: {:?}", available_stt);
    println!("Available TTS providers: {:?}", available_tts);
    
    if available_stt.is_empty() && available_tts.is_empty() {
        println!("\nNo providers were registered. This suggests the provider configurations were not loaded from environment.");
    }
    
    Ok(())
}