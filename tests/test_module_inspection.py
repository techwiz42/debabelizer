#!/usr/bin/env python3

import os
import sys

# Set up environment for Soniox
os.environ["SONIOX_API_KEY"] = "fake_key_for_testing"

try:
    print("🔍 Importing debabelizer...")
    import debabelizer
    print("✅ Successfully imported debabelizer")
    
    print(f"\n🔍 debabelizer module path: {debabelizer.__file__}")
    print(f"🔍 debabelizer module attributes: {dir(debabelizer)}")
    
    # Check if VoiceProcessor is available
    if hasattr(debabelizer, 'VoiceProcessor'):
        print(f"✅ VoiceProcessor found: {debabelizer.VoiceProcessor}")
        print(f"🔍 VoiceProcessor type: {type(debabelizer.VoiceProcessor)}")
        print(f"🔍 VoiceProcessor module: {debabelizer.VoiceProcessor.__module__}")
    else:
        print("❌ VoiceProcessor not found in debabelizer module")
    
    # Check for other classes
    if hasattr(debabelizer, 'AudioData'):
        print(f"✅ AudioData found: {debabelizer.AudioData}")
    else:
        print("❌ AudioData not found - likely legacy Python implementation")
    
    if hasattr(debabelizer, 'AudioFormat'):
        print(f"✅ AudioFormat found: {debabelizer.AudioFormat}")
    else:
        print("❌ AudioFormat not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()