#!/usr/bin/env python3
"""
Advanced usage example for Debabelizer Python bindings.

This example demonstrates:
1. Provider selection strategies
2. Error handling and recovery
3. Batch processing
4. Performance optimization
5. Custom configuration scenarios
"""

import debabelizer
from debabelizer.utils import create_config_from_env, create_synthesis_options
import time
import os
from typing import List, Dict, Any


class AdvancedDebabelizer:
    """Advanced wrapper for Debabelizer functionality."""
    
    def __init__(self):
        self.processors: Dict[str, debabelizer.VoiceProcessor] = {}
        self.setup_multiple_processors()
    
    def setup_multiple_processors(self):
        """Set up multiple processors with different provider configurations."""
        
        # Configuration for different use cases
        configs = {
            "high_quality": {
                "preferences": {
                    "optimize_for": "quality",
                    "stt_provider": "deepgram",
                    "tts_provider": "elevenlabs"
                }
            },
            "low_latency": {
                "preferences": {
                    "optimize_for": "latency", 
                    "stt_provider": "soniox",
                    "tts_provider": "openai"
                }
            },
            "cost_effective": {
                "preferences": {
                    "optimize_for": "cost",
                    "stt_provider": "whisper",  # Local processing
                    "tts_provider": "openai"
                }
            }
        }
        
        for name, config in configs.items():
            try:
                # Merge with environment configuration
                full_config = create_config_from_env()
                full_config.update(config)
                
                processor = debabelizer.VoiceProcessor(config=full_config)
                self.processors[name] = processor
                print(f"✓ Created {name} processor: STT={processor.get_stt_provider_name()}, TTS={processor.get_tts_provider_name()}")
                
            except debabelizer.DebabelizerException as e:
                print(f"✗ Failed to create {name} processor: {e}")
    
    def transcribe_with_fallback(self, audio: debabelizer.AudioData, preferred_quality: str = "high_quality") -> debabelizer.TranscriptionResult:
        """Transcribe audio with automatic fallback to other providers."""
        
        # Try processors in order of preference
        fallback_order = [preferred_quality, "low_latency", "cost_effective"]
        
        for processor_name in fallback_order:
            if processor_name not in self.processors:
                continue
                
            processor = self.processors[processor_name]
            if not processor.has_stt_provider():
                continue
                
            try:
                print(f"Attempting transcription with {processor_name} processor...")
                start_time = time.time()
                
                result = processor.transcribe(audio)
                
                elapsed_time = time.time() - start_time
                print(f"✓ Transcription successful with {processor.get_stt_provider_name()} in {elapsed_time:.2f}s")
                
                return result
                
            except debabelizer.DebabelizerException as e:
                print(f"✗ Transcription failed with {processor_name}: {e}")
                continue
        
        raise debabelizer.DebabelizerException("All transcription providers failed")
    
    def synthesize_with_options(self, text: str, voice_style: str = "default") -> debabelizer.SynthesisResult:
        """Synthesize speech with different voice styles."""
        
        voice_styles = {
            "default": create_synthesis_options(voice="alloy", speed=1.0, format="mp3"),
            "fast": create_synthesis_options(voice="nova", speed=1.5, format="mp3"),
            "slow": create_synthesis_options(voice="echo", speed=0.8, format="mp3"),
            "deep": create_synthesis_options(voice="onyx", speed=0.9, pitch=-5.0, format="mp3"),
            "high": create_synthesis_options(voice="shimmer", speed=1.1, pitch=2.0, format="mp3"),
        }
        
        options = voice_styles.get(voice_style, voice_styles["default"])
        
        # Try high quality first, fall back to others
        for processor_name in ["high_quality", "low_latency", "cost_effective"]:
            if processor_name not in self.processors:
                continue
                
            processor = self.processors[processor_name]
            if not processor.has_tts_provider():
                continue
                
            try:
                print(f"Synthesizing with {processor_name} processor using {voice_style} style...")
                result = processor.synthesize(text, options)
                print(f"✓ Synthesis successful with {processor.get_tts_provider_name()}")
                return result
                
            except debabelizer.DebabelizerException as e:
                print(f"✗ Synthesis failed with {processor_name}: {e}")
                continue
        
        raise debabelizer.DebabelizerException("All synthesis providers failed")
    
    def batch_transcribe(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio files with error handling."""
        
        results = []
        
        for i, file_path in enumerate(audio_files):
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {file_path}")
            
            try:
                # Load audio file
                with open(file_path, "rb") as f:
                    audio_data = f.read()
                
                # Determine format from extension
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                
                # Create audio format (adjust these based on your files)
                audio_format = debabelizer.AudioFormat(
                    format=ext,
                    sample_rate=16000,
                    channels=1,
                    bit_depth=16
                )
                
                audio = debabelizer.AudioData(audio_data, audio_format)
                
                # Transcribe with fallback
                result = self.transcribe_with_fallback(audio)
                
                results.append({
                    "file": file_path,
                    "status": "success",
                    "text": result.text,
                    "confidence": result.confidence,
                    "language": result.language_detected,
                    "duration": result.duration,
                    "words": len(result.words) if result.words else 0
                })
                
            except Exception as e:
                print(f"✗ Failed to process {file_path}: {e}")
                results.append({
                    "file": file_path,
                    "status": "error", 
                    "error": str(e)
                })
        
        return results
    
    def voice_analysis(self):
        """Analyze available voices across all providers."""
        
        print("\n=== Voice Analysis ===")
        
        all_voices = {}
        
        for processor_name, processor in self.processors.items():
            if not processor.has_tts_provider():
                continue
                
            try:
                voices = processor.get_available_voices()
                provider_name = processor.get_tts_provider_name()
                all_voices[f"{processor_name} ({provider_name})"] = voices
                
                print(f"\n{processor_name} ({provider_name}): {len(voices)} voices")
                
                # Analyze by language
                languages = {}
                for voice in voices:
                    lang = voice.language
                    if lang not in languages:
                        languages[lang] = []
                    languages[lang].append(voice)
                
                print(f"  Languages supported: {len(languages)}")
                for lang, lang_voices in sorted(languages.items()):
                    print(f"    {lang}: {len(lang_voices)} voices")
                
            except debabelizer.DebabelizerException as e:
                print(f"Failed to get voices for {processor_name}: {e}")
        
        return all_voices
    
    def performance_benchmark(self, test_text: str = "This is a performance test of the Debabelizer library."):
        """Benchmark different providers for performance."""
        
        print("\n=== Performance Benchmark ===")
        
        results = {}
        
        for processor_name, processor in self.processors.items():
            if not processor.has_tts_provider():
                continue
                
            try:
                print(f"\nBenchmarking {processor_name}...")
                
                # Time synthesis
                start_time = time.time()
                result = processor.synthesize(test_text)
                synthesis_time = time.time() - start_time
                
                results[processor_name] = {
                    "provider": processor.get_tts_provider_name(),
                    "synthesis_time": synthesis_time,
                    "audio_size": result.size_bytes,
                    "duration": result.duration,
                    "format": result.format
                }
                
                print(f"  Time: {synthesis_time:.3f}s")
                print(f"  Size: {result.size_bytes} bytes")
                print(f"  Duration: {result.duration}s")
                
            except debabelizer.DebabelizerException as e:
                print(f"  Failed: {e}")
                results[processor_name] = {"error": str(e)}
        
        return results


def main():
    """Main example function."""
    
    print("=== Advanced Debabelizer Usage Example ===")
    
    # Create advanced wrapper
    advanced = AdvancedDebabelizer()
    
    if not advanced.processors:
        print("No processors could be created. Check your API key configuration.")
        return
    
    # Example 1: Voice style synthesis
    print("\n=== Voice Style Synthesis ===")
    
    test_text = "The quick brown fox jumps over the lazy dog. This is a test of different voice styles."
    
    styles = ["default", "fast", "slow", "deep", "high"]
    
    for style in styles:
        try:
            result = advanced.synthesize_with_options(test_text, style)
            filename = f"voice_style_{style}.mp3"
            with open(filename, "wb") as f:
                f.write(result.audio_data)
            print(f"✓ Created {filename} with {style} style")
        except debabelizer.DebabelizerException as e:
            print(f"✗ Failed to create {style} style: {e}")
    
    # Example 2: Batch processing (if you have audio files)
    print("\n=== Batch Processing ===")
    
    # Create some dummy audio files for demonstration
    audio_files = ["audio1.wav", "audio2.mp3", "audio3.flac"]
    
    # In real usage, you would have actual audio files
    print("Batch processing would process these files:")
    for file in audio_files:
        print(f"  - {file}")
    
    # Uncomment to run with real files:
    # results = advanced.batch_transcribe(audio_files)
    # for result in results:
    #     print(f"  {result['file']}: {result['status']}")
    
    # Example 3: Voice analysis
    voices_data = advanced.voice_analysis()
    
    # Example 4: Performance benchmark
    benchmark_results = advanced.performance_benchmark()
    
    print("\n=== Benchmark Results Summary ===")
    for processor_name, result in benchmark_results.items():
        if "error" not in result:
            print(f"{processor_name}: {result['synthesis_time']:.3f}s ({result['provider']})")
    
    # Example 5: Error handling and recovery
    print("\n=== Error Handling Example ===")
    
    # Create invalid audio data to test error handling
    invalid_audio = debabelizer.AudioData(
        b"invalid_audio_data",
        debabelizer.AudioFormat("wav", 16000, 1, 16)
    )
    
    try:
        result = advanced.transcribe_with_fallback(invalid_audio)
        print(f"Unexpected success: {result.text}")
    except debabelizer.DebabelizerException as e:
        print(f"Expected error caught: {e}")
    
    print("\n=== Advanced Example Complete ===")


if __name__ == "__main__":
    main()