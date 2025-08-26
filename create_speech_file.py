#!/usr/bin/env python3
"""Create a speech audio file using gTTS or pyttsx3."""

import os
import sys

try:
    # Try using pyttsx3 first (offline TTS)
    import pyttsx3
    
    def create_with_pyttsx3():
        engine = pyttsx3.init()
        text = "Hello, this is a test of the speech recognition system. The quick brown fox jumps over the lazy dog."
        
        # Set properties
        engine.setProperty('rate', 150)  # Speed of speech
        
        # Save to file
        engine.save_to_file(text, 'test_speech.wav')
        engine.runAndWait()
        
        print("Created test_speech.wav using pyttsx3")
        return True
        
    if create_with_pyttsx3():
        sys.exit(0)
        
except ImportError:
    print("pyttsx3 not available")

try:
    # Try gTTS (requires internet)
    from gtts import gTTS
    
    def create_with_gtts():
        text = "Hello, this is a test of the speech recognition system. The quick brown fox jumps over the lazy dog."
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("test_speech.mp3")
        
        # Convert to WAV using pydub if available
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3("test_speech.mp3")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export("test_speech.wav", format="wav")
            os.remove("test_speech.mp3")
            print("Created test_speech.wav using gTTS and pydub")
        except ImportError:
            print("Created test_speech.mp3 using gTTS (install pydub to convert to WAV)")
        return True
        
    if create_with_gtts():
        sys.exit(0)
        
except ImportError:
    print("gTTS not available")

# If no TTS libraries available, create using espeak command
if os.system("which espeak > /dev/null 2>&1") == 0:
    text = "Hello, this is a test of the speech recognition system. The quick brown fox jumps over the lazy dog."
    cmd = f'espeak -w test_speech_raw.wav "{text}"'
    if os.system(cmd) == 0:
        # Convert to 16kHz mono using sox or ffmpeg
        if os.system("which sox > /dev/null 2>&1") == 0:
            os.system("sox test_speech_raw.wav -r 16000 -c 1 test_speech.wav")
            os.remove("test_speech_raw.wav")
            print("Created test_speech.wav using espeak and sox")
        elif os.system("which ffmpeg > /dev/null 2>&1") == 0:
            os.system("ffmpeg -i test_speech_raw.wav -ar 16000 -ac 1 test_speech.wav -y")
            os.remove("test_speech_raw.wav") 
            print("Created test_speech.wav using espeak and ffmpeg")
        else:
            os.rename("test_speech_raw.wav", "test_speech.wav")
            print("Created test_speech.wav using espeak (may not be 16kHz)")
        sys.exit(0)

print("No TTS method available. Please install pyttsx3, gtts, or espeak")