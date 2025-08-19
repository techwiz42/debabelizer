//! Performance and load testing for Debabelizer

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Semaphore;

    #[tokio::test]
    async fn test_processor_creation_performance() {
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let processor = VoiceProcessor::new();
            assert!(processor.is_ok());
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_millis() as f64 / iterations as f64;
        
        // Should create processors quickly (less than 50ms average)
        assert!(avg_time < 50.0, "Processor creation too slow: {}ms average", avg_time);
        
        println!("Processor creation: {}ms average over {} iterations", avg_time, iterations);
    }

    #[tokio::test]
    async fn test_concurrent_processor_creation() {
        let concurrent_count = 50;
        let start = Instant::now();
        
        let mut handles = vec![];
        for _ in 0..concurrent_count {
            let handle = tokio::spawn(async {
                VoiceProcessor::new()
            });
            handles.push(handle);
        }
        
        let results = futures::future::join_all(handles).await;
        let duration = start.elapsed();
        
        // All should succeed
        for result in results {
            assert!(result.unwrap().is_ok());
        }
        
        let avg_time = duration.as_millis() as f64 / concurrent_count as f64;
        println!("Concurrent processor creation: {}ms average for {} processors", avg_time, concurrent_count);
        
        // Should handle concurrent creation efficiently
        assert!(avg_time < 100.0, "Concurrent creation too slow: {}ms average", avg_time);
    }

    #[tokio::test]
    async fn test_session_manager_performance() {
        let manager = SessionManager::new();
        let session_count = 1000;
        let start = Instant::now();
        
        // Create many sessions
        let mut session_ids = vec![];
        for _ in 0..session_count {
            let session = manager.create_session().await;
            session_ids.push(session.id);
        }
        
        let creation_time = start.elapsed();
        
        // Update all sessions
        let update_start = Instant::now();
        for (i, session_id) in session_ids.iter().enumerate() {
            manager.update_session(*session_id, |s| {
                s.metadata.insert("index".to_string(), serde_json::json!(i));
            }).await.unwrap();
        }
        let update_time = update_start.elapsed();
        
        // List all sessions
        let list_start = Instant::now();
        let sessions = manager.list_sessions().await;
        let list_time = list_start.elapsed();
        
        assert_eq!(sessions.len(), session_count);
        
        println!("Session performance for {} sessions:", session_count);
        println!("  Creation: {}ms", creation_time.as_millis());
        println!("  Updates: {}ms", update_time.as_millis());
        println!("  Listing: {}ms", list_time.as_millis());
        
        // Performance thresholds
        assert!(creation_time.as_millis() < 1000, "Session creation too slow");
        assert!(update_time.as_millis() < 2000, "Session updates too slow");
        assert!(list_time.as_millis() < 100, "Session listing too slow");
    }

    #[tokio::test]
    async fn test_concurrent_session_operations() {
        let manager = Arc::new(SessionManager::new());
        let concurrent_operations = 100;
        let operations_per_task = 10;
        
        let start = Instant::now();
        let mut handles = vec![];
        
        for task_id in 0..concurrent_operations {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                let mut local_sessions = vec![];
                
                // Create sessions
                for i in 0..operations_per_task {
                    let session = manager_clone.create_session().await;
                    local_sessions.push(session.id);
                    
                    // Update session
                    manager_clone.update_session(session.id, |s| {
                        s.metadata.insert("task".to_string(), serde_json::json!(task_id));
                        s.metadata.insert("operation".to_string(), serde_json::json!(i));
                    }).await.unwrap();
                }
                
                // Read back sessions
                for session_id in &local_sessions {
                    let session = manager_clone.get_session(*session_id).await;
                    assert!(session.is_some());
                }
                
                local_sessions.len()
            });
            handles.push(handle);
        }
        
        let results = futures::future::join_all(handles).await;
        let duration = start.elapsed();
        
        let total_operations: usize = results.into_iter().map(|r| r.unwrap()).sum();
        let expected_operations = concurrent_operations * operations_per_task;
        
        assert_eq!(total_operations, expected_operations);
        
        println!("Concurrent session operations:");
        println!("  {} tasks with {} ops each = {} total operations", 
                 concurrent_operations, operations_per_task, total_operations);
        println!("  Total time: {}ms", duration.as_millis());
        println!("  Ops/second: {:.1}", total_operations as f64 / duration.as_secs_f64());
        
        // Should handle high concurrency efficiently
        assert!(duration.as_secs() < 10, "Concurrent operations too slow");
    }

    #[tokio::test]
    async fn test_config_parsing_performance() {
        let iterations = 1000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let config = DebabelizerConfig::default();
            
            // Test various config operations
            let _ = config.is_auto_select_enabled();
            let _ = config.get_optimization_strategy();
            let _ = config.get_preferred_stt_provider();
            let _ = config.get_preferred_tts_provider();
            let _ = config.get_provider_config("nonexistent");
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_micros() as f64 / iterations as f64;
        
        println!("Config operations: {:.1}μs average over {} iterations", avg_time, iterations);
        
        // Should be very fast (less than 100 microseconds)
        assert!(avg_time < 100.0, "Config operations too slow: {:.1}μs average", avg_time);
    }

    #[tokio::test]
    async fn test_audio_format_detection_performance() {
        use debabelizer_utils::audio::detect_audio_format;
        
        // Create test audio headers
        let wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xAC\x00\x00";
        let mp3_header = b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\xFF\xFB\x90\x00";
        let ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        
        let iterations = 10000;
        let start = Instant::now();
        
        for i in 0..iterations {
            let header: &[u8] = match i % 3 {
                0 => wav_header,
                1 => mp3_header,
                _ => ogg_header,
            };
            
            let result = detect_audio_format(header);
            assert!(result.is_ok());
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        
        println!("Audio format detection: {:.1}ns average over {} iterations", avg_time, iterations);
        
        // Should be extremely fast (less than 10 microseconds)
        assert!(avg_time < 10_000.0, "Format detection too slow: {:.1}ns average", avg_time);
    }

    #[tokio::test]
    async fn test_memory_usage_stability() {
        // Test that repeated operations don't cause memory leaks
        let processor = VoiceProcessor::new().unwrap();
        let manager = SessionManager::new();
        
        // Perform many operations in a loop
        for iteration in 0..100 {
            // Create and destroy sessions
            let session = manager.create_session().await;
            manager.update_session(session.id, |s| {
                s.metadata.insert("iteration".to_string(), serde_json::json!(iteration));
            }).await.unwrap();
            manager.remove_session(session.id).await;
            
            // Create audio data
            let audio_data = vec![0u8; 1000];
            let audio = AudioData::new(audio_data, AudioFormat::wav(16000));
            
            // Test provider listing
            let _ = processor.list_stt_providers().await;
            let _ = processor.list_tts_providers().await;
            
            // Simulate processing (will fail without API keys, but tests memory usage)
            let _ = processor.transcribe(audio).await;
        }
        
        // If we reach here without OOM, memory usage is stable
        println!("Memory stability test completed 100 iterations successfully");
    }

    #[tokio::test]
    async fn test_rate_limited_operations() {
        // Test behavior under rate limiting simulation
        let manager = Arc::new(SessionManager::new());
        let semaphore = Arc::new(Semaphore::new(10)); // Limit to 10 concurrent operations
        let total_operations = 100;
        
        let start = Instant::now();
        let mut handles = vec![];
        
        for i in 0..total_operations {
            let manager_clone = manager.clone();
            let semaphore_clone = semaphore.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                
                // Simulate some work
                tokio::time::sleep(Duration::from_millis(10)).await;
                
                let session = manager_clone.create_session().await;
                manager_clone.update_session(session.id, |s| {
                    s.metadata.insert("operation".to_string(), serde_json::json!(i));
                }).await.unwrap();
                
                session.id
            });
            handles.push(handle);
        }
        
        let results = futures::future::join_all(handles).await;
        let duration = start.elapsed();
        
        assert_eq!(results.len(), total_operations);
        for result in results {
            assert!(result.is_ok());
        }
        
        println!("Rate limited operations:");
        println!("  {} operations with max 10 concurrent", total_operations);
        println!("  Total time: {}ms", duration.as_millis());
        
        // Should complete within reasonable time even with rate limiting
        assert!(duration.as_secs() < 30, "Rate limited operations took too long");
    }

    #[tokio::test]
    async fn test_large_data_handling() {
        // Test handling of large audio data
        let sizes = vec![
            1_000,      // 1KB
            10_000,     // 10KB
            100_000,    // 100KB
            1_000_000,  // 1MB
        ];
        
        for size in sizes {
            let start = Instant::now();
            
            // Create large audio data
            let audio_data = vec![0u8; size];
            let audio = AudioData::new(audio_data, AudioFormat::wav(16000));
            
            // Test creation time
            let creation_time = start.elapsed();
            
            // Test format access
            let access_start = Instant::now();
            let _format = &audio.format;
            let _data_len = audio.data.len();
            let access_time = access_start.elapsed();
            
            println!("Large data handling for {} bytes:", size);
            println!("  Creation: {}μs", creation_time.as_micros());
            println!("  Access: {}μs", access_time.as_micros());
            
            // Should handle large data efficiently
            assert!(creation_time.as_millis() < 100, "Large data creation too slow");
            assert!(access_time.as_micros() < 1000, "Large data access too slow");
        }
    }

    #[test]
    fn test_serialization_performance() {
        use serde_json;
        
        // Test JSON serialization performance for various data structures
        let iterations = 1000;
        
        // Test transcription result serialization
        let transcription = TranscriptionResult {
            text: "Hello world ".repeat(100),
            confidence: 0.95,
            language_detected: Some("en".to_string()),
            duration: Some(5.0),
            words: Some((0..100).map(|i| WordTiming {
                word: format!("word{}", i),
                start: i as f32 * 0.1,
                end: (i + 1) as f32 * 0.1,
                confidence: 0.9 + (i as f32 / 1000.0),
            }).collect()),
            metadata: None,
        };
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = serde_json::to_string(&transcription).unwrap();
        }
        let serialization_time = start.elapsed();
        
        // Test deserialization
        let json = serde_json::to_string(&transcription).unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            let _result: TranscriptionResult = serde_json::from_str(&json).unwrap();
        }
        let deserialization_time = start.elapsed();
        
        println!("JSON serialization performance:");
        println!("  Serialization: {}μs average", serialization_time.as_micros() / iterations);
        println!("  Deserialization: {}μs average", deserialization_time.as_micros() / iterations);
        
        // Should be reasonably fast
        assert!(serialization_time.as_millis() < 1000, "Serialization too slow");
        assert!(deserialization_time.as_millis() < 1000, "Deserialization too slow");
    }

    #[tokio::test]
    async fn test_stress_provider_listing() {
        // Stress test provider listing operations
        let processor = VoiceProcessor::new().unwrap();
        let iterations = 1000;
        
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _stt_providers = processor.list_stt_providers().await.unwrap();
            let _tts_providers = processor.list_tts_providers().await.unwrap();
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_micros() as f64 / (iterations * 2) as f64;
        
        println!("Provider listing stress test:");
        println!("  {} iterations of both STT and TTS listing", iterations);
        println!("  Average time per operation: {:.1}μs", avg_time);
        
        // Should handle repeated listing efficiently
        assert!(avg_time < 1000.0, "Provider listing too slow under stress: {:.1}μs", avg_time);
    }
}