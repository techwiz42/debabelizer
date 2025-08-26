#!/usr/bin/env python3
"""
Debug audio chunk size calculation for Soniox streaming
"""

def analyze_audio_chunk(chunk_size_bytes, sample_rate=16000, channels=1, bits_per_sample=16):
    """Calculate audio chunk duration and characteristics"""
    
    # PCM audio calculations
    bytes_per_sample = bits_per_sample // 8  # 16-bit = 2 bytes
    bytes_per_second = sample_rate * channels * bytes_per_sample
    
    # Calculate duration
    duration_seconds = chunk_size_bytes / bytes_per_second
    duration_ms = duration_seconds * 1000
    
    # Calculate samples
    samples = chunk_size_bytes // bytes_per_sample
    
    print(f"🎵 Audio Chunk Analysis:")
    print(f"   Chunk Size: {chunk_size_bytes} bytes")
    print(f"   Sample Rate: {sample_rate} Hz")
    print(f"   Channels: {channels}")
    print(f"   Bits per Sample: {bits_per_sample}")
    print(f"   Bytes per Sample: {bytes_per_sample}")
    print(f"   Bytes per Second: {bytes_per_second}")
    print(f"   Duration: {duration_ms:.2f} ms ({duration_seconds:.4f} seconds)")
    print(f"   Number of Samples: {samples}")
    
    return duration_ms

def main():
    print("="*60)
    print("🔍 Analyzing current 170-byte chunks:")
    current_duration = analyze_audio_chunk(170)
    
    print("\n" + "="*60)
    print("📊 Recommended chunk sizes for real-time streaming:")
    
    # Common real-time audio chunk durations
    recommended_durations = [10, 20, 50, 100, 200]  # milliseconds
    
    for duration_ms in recommended_durations:
        chunk_size = int((duration_ms / 1000) * 16000 * 2)  # 16kHz, 16-bit
        print(f"   {duration_ms:3d}ms = {chunk_size:4d} bytes")
    
    print("\n" + "="*60)
    print("🎯 Analysis:")
    
    if current_duration < 10:
        print(f"❌ Current chunks ({current_duration:.2f}ms) are TOO SMALL!")
        print("   🔧 Soniox may need longer audio segments to detect speech")
        print("   💡 Recommended: 20-50ms chunks (640-1600 bytes)")
    elif current_duration > 200:
        print(f"⚠️  Current chunks ({current_duration:.2f}ms) might be too large for real-time")
        print("   💡 Recommended: 20-50ms chunks for better latency")
    else:
        print(f"✅ Current chunk size ({current_duration:.2f}ms) looks reasonable")
        
    print(f"\n🎤 For comparison:")
    print(f"   - Phone call quality: ~20ms chunks")
    print(f"   - Video conferencing: ~20-40ms chunks") 
    print(f"   - High-quality streaming: ~50-100ms chunks")

if __name__ == "__main__":
    main()