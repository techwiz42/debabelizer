// Simple test to debug Rust configuration and provider loading
use std::collections::HashMap;
use serde_json::json;

// Manually import the exact path
#[path = "debabelizer/src/config.rs"]
mod config;

#[path = "debabelizer/src/providers.rs"] 
mod providers;

use config::DebabelizerConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debugging Rust Soniox Configuration and Provider Loading");
    println!("={}", "=".repeat(60));
    
    // Set the environment variable
    std::env::set_var("SONIOX_API_KEY", "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95");
    println!("✅ Set SONIOX_API_KEY environment variable");
    
    // Test 1: Check if config loads environment variables
    println!("\n📋 Testing configuration loading...");
    let config = match DebabelizerConfig::from_env() {
        Ok(c) => {
            println!("✅ Configuration loaded successfully");
            c
        }
        Err(e) => {
            println!("❌ Configuration loading failed: {}", e);
            println!("📝 Creating manual config as fallback...");
            
            // Manual config creation
            let mut providers = HashMap::new();
            let mut soniox_config = HashMap::new();
            soniox_config.insert("api_key".to_string(), json!("cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"));
            soniox_config.insert("model".to_string(), json!("stt-rt-preview-v2"));
            soniox_config.insert("auto_detect_language".to_string(), json!(true));
            
            providers.insert("soniox".to_string(), config::ProviderConfig::Simple(soniox_config));
            
            DebabelizerConfig {
                preferences: config::Preferences {
                    stt_provider: Some("soniox".to_string()),
                    tts_provider: None,
                    auto_select: false,
                    optimize_for: config::OptimizationStrategy::Balanced,
                    timeout_seconds: None,
                    max_retries: None,
                },
                providers,
            }
        }
    };
    
    println!("  📊 Config details:");
    println!("    - Preferred STT: {:?}", config.get_preferred_stt_provider());
    println!("    - Preferred TTS: {:?}", config.get_preferred_tts_provider());
    println!("    - Provider count: {}", config.providers.len());
    
    // Test 2: Check Soniox provider config specifically
    println!("\n🔍 Testing Soniox provider configuration...");
    if let Some(soniox_config) = config.get_provider_config("soniox") {
        println!("✅ Found Soniox provider config");
        if let Some(api_key) = soniox_config.get_api_key() {
            println!("  ✅ API key found: {}***", &api_key[..6]);
        } else {
            println!("  ❌ No API key found in Soniox config");
        }
    } else {
        println!("❌ No Soniox provider config found");
        println!("  📝 Available providers: {:?}", config.providers.keys().collect::<Vec<_>>());
    }
    
    // Test 3: Initialize providers 
    println!("\n🚀 Testing provider initialization...");
    match providers::initialize_providers(&config).await {
        Ok(registry) => {
            println!("✅ Provider registry initialized");
            let stt_providers = registry.list_stt_providers();
            let tts_providers = registry.list_tts_providers();
            
            println!("  📊 STT providers available: {:?}", stt_providers);
            println!("  📊 TTS providers available: {:?}", tts_providers);
            
            if stt_providers.contains(&"soniox".to_string()) {
                println!("  ✅ Soniox STT provider is registered!");
                
                // Test 4: Get the Soniox provider directly
                if let Some(soniox_provider) = registry.get_stt("soniox") {
                    println!("  ✅ Retrieved Soniox provider successfully");
                    println!("    - Name: {}", soniox_provider.name());
                    println!("    - Supports streaming: {}", soniox_provider.supports_streaming());
                    println!("    - Supported formats: {:?}", soniox_provider.supported_formats());
                    
                    // Test 5: Try to create a streaming session
                    println!("\n🌊 Testing stream creation...");
                    let stream_config = debabelizer_core::StreamConfig {
                        session_id: uuid::Uuid::new_v4(),
                        format: debabelizer_core::AudioFormat {
                            format: "wav".to_string(),
                            sample_rate: 16000,
                            channels: 1,
                            bit_depth: Some(16),
                        },
                        interim_results: true,
                        language: None,
                        model: None,
                        punctuate: true,
                        profanity_filter: false,
                        diarization: false,
                        metadata: None,
                        enable_word_time_offsets: true,
                        enable_automatic_punctuation: true,
                        enable_language_identification: true,
                    };
                    
                    println!("  📋 Stream config: session_id = {}", stream_config.session_id);
                    
                    match soniox_provider.transcribe_stream(stream_config).await {
                        Ok(mut stream) => {
                            println!("  ✅ Stream created successfully!");
                            println!("    - Session ID: {}", stream.session_id());
                            
                            // Test receiving a result (should timeout if no audio)
                            println!("  🔍 Testing result reception (with timeout)...");
                            match tokio::time::timeout(
                                std::time::Duration::from_secs(2),
                                stream.receive_transcript()
                            ).await {
                                Ok(Ok(Some(result))) => {
                                    println!("    ✅ Received result: '{}'", result.text);
                                }
                                Ok(Ok(None)) => {
                                    println!("    ⚠️ Stream ended (no audio sent)");
                                }
                                Ok(Err(e)) => {
                                    println!("    ❌ Error receiving result: {:?}", e);
                                }
                                Err(_timeout) => {
                                    println!("    ⏰ Timeout (expected - no audio sent)");
                                }
                            }
                            
                            // Close the stream
                            if let Err(e) = stream.close().await {
                                println!("    ⚠️ Error closing stream: {:?}", e);
                            } else {
                                println!("    ✅ Stream closed successfully");
                            }
                        }
                        Err(e) => {
                            println!("  ❌ Failed to create stream: {:?}", e);
                        }
                    }
                    
                } else {
                    println!("  ❌ Failed to retrieve Soniox provider from registry");
                }
            } else {
                println!("  ❌ Soniox STT provider is NOT registered");
                println!("    This indicates the provider creation failed");
            }
        }
        Err(e) => {
            println!("❌ Provider initialization failed: {:?}", e);
        }
    }
    
    println!("\n🎯 Summary:");
    println!("- This test isolates each step of the provider loading process");
    println!("- If Soniox provider is registered but stream creation fails,");
    println!("  the issue is in the WebSocket connection logic");
    println!("- If Soniox provider is NOT registered, the issue is in");
    println!("  the provider initialization or configuration");
    
    println!("\n🏁 Rust configuration debug test completed!");
    println!("={}", "=".repeat(60));
    
    Ok(())
}