use debabelizer_core::{AudioFormat, DebabelizerError, Result};

pub fn detect_audio_format(data: &[u8]) -> Result<AudioFormat> {
    if data.len() < 12 {
        return Err(DebabelizerError::AudioFormat(
            "Not enough data to detect format".to_string(),
        ));
    }
    
    // WAV detection (RIFF header)
    if &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        // Try to extract sample rate from WAV header
        if data.len() >= 28 {
            let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
            return Ok(AudioFormat::wav(sample_rate));
        }
        return Ok(AudioFormat::wav(16000));
    }
    
    // MP3 detection (ID3 tag or frame sync)
    if &data[0..3] == b"ID3" || (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
        return Ok(AudioFormat::mp3(16000));
    }
    
    // OGG/Opus detection
    if &data[0..4] == b"OggS" {
        return Ok(AudioFormat::opus(48000));
    }
    
    // FLAC detection
    if &data[0..4] == b"fLaC" {
        return Ok(AudioFormat {
            format: "flac".to_string(),
            sample_rate: 16000,
            channels: 1,
            bit_depth: Some(16),
        });
    }
    
    Err(DebabelizerError::AudioFormat(
        "Unknown audio format".to_string(),
    ))
}

pub fn calculate_duration_seconds(data: &[u8], format: &AudioFormat) -> Option<f32> {
    match format.format.as_str() {
        "wav" => {
            // For WAV, calculate from data size and format
            if data.len() > 44 {
                let data_size = data.len() - 44; // Subtract header
                let bytes_per_sample = format.bit_depth.unwrap_or(16) as usize / 8;
                let bytes_per_second = format.sample_rate as usize * format.channels as usize * bytes_per_sample;
                return Some(data_size as f32 / bytes_per_second as f32);
            }
        }
        _ => {
            // For compressed formats, we'd need to decode to get accurate duration
            // This is a placeholder - real implementation would use symphonia
            return None;
        }
    }
    None
}

pub fn convert_sample_rate(
    data: &[u8],
    from_format: &AudioFormat,
    target_sample_rate: u32,
) -> Result<Vec<u8>> {
    if from_format.sample_rate == target_sample_rate {
        return Ok(data.to_vec());
    }
    
    // This is a simplified placeholder - real implementation would use proper resampling
    Err(DebabelizerError::AudioFormat(
        "Sample rate conversion not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wav_detection() {
        let wav_header = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        let format = detect_audio_format(wav_header).unwrap();
        assert_eq!(format.format, "wav");
    }
    
    #[test]
    fn test_mp3_detection() {
        let mp3_header = b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00";
        let format = detect_audio_format(mp3_header).unwrap();
        assert_eq!(format.format, "mp3");
    }
    
    #[test]
    fn test_unknown_format() {
        let unknown = b"UNKNOWN FORMAT";
        let result = detect_audio_format(unknown);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let short_data = b"ABC";
        let result = detect_audio_format(short_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not enough data"));
    }

    #[test]
    fn test_wav_with_sample_rate() {
        // WAV header with 44100 Hz sample rate
        let mut wav_header = vec![0u8; 28];
        wav_header[0..4].copy_from_slice(b"RIFF");
        wav_header[8..12].copy_from_slice(b"WAVE");
        // Sample rate: 44100 (0xAC44) in little endian
        wav_header[24..28].copy_from_slice(&44100u32.to_le_bytes());
        
        let format = detect_audio_format(&wav_header).unwrap();
        assert_eq!(format.format, "wav");
        assert_eq!(format.sample_rate, 44100);
    }

    #[test]
    fn test_mp3_frame_sync() {
        // MP3 frame sync pattern
        let mp3_sync = b"\xFF\xFB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let format = detect_audio_format(mp3_sync).unwrap();
        assert_eq!(format.format, "mp3");
    }

    #[test]
    fn test_ogg_detection() {
        let ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        let format = detect_audio_format(ogg_header).unwrap();
        assert_eq!(format.format, "opus");
        assert_eq!(format.sample_rate, 48000);
    }

    #[test]
    fn test_flac_detection() {
        let flac_header = b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00";
        let format = detect_audio_format(flac_header).unwrap();
        assert_eq!(format.format, "flac");
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.bit_depth, Some(16));
    }

    #[test]
    fn test_calculate_duration_wav() {
        let wav_format = AudioFormat::wav(16000);
        // 44-byte header + 16000 bytes of audio data = 1 second at 16kHz mono 16-bit
        let wav_data = vec![0u8; 44 + 32000]; // 32000 bytes = 16000 samples * 2 bytes
        
        let duration = calculate_duration_seconds(&wav_data, &wav_format);
        assert!(duration.is_some());
        let duration = duration.unwrap();
        assert!((duration - 1.0).abs() < 0.1); // Should be approximately 1 second
    }

    #[test]
    fn test_calculate_duration_unknown_format() {
        let unknown_format = AudioFormat {
            format: "unknown".to_string(),
            sample_rate: 16000,
            channels: 1,
            bit_depth: Some(16),
        };
        let data = vec![0u8; 1000];
        
        let duration = calculate_duration_seconds(&data, &unknown_format);
        assert!(duration.is_none());
    }

    #[test]
    fn test_calculate_duration_wav_too_short() {
        let wav_format = AudioFormat::wav(16000);
        let short_data = vec![0u8; 40]; // Less than 44 byte header
        
        let duration = calculate_duration_seconds(&short_data, &wav_format);
        assert!(duration.is_none());
    }
}