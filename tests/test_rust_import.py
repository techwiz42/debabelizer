#!/usr/bin/env python3

import sys
import os

# Add the path where our compiled Rust library is located  
target_path = "/home/peter/debabelizer/target/release"
if target_path not in sys.path:
    sys.path.insert(0, target_path)

print("🔍 Attempting to import Rust compiled library...")
print(f"Python path includes: {target_path}")
print(f"Available files: {os.listdir(target_path)}")

try:
    # Try to import the compiled library directly
    import lib_internal as rust_debabelizer
    print("✅ Successfully imported lib_internal!")
    print(f"Available attributes: {dir(rust_debabelizer)}")
    
    # Try to create a VoiceProcessor
    if hasattr(rust_debabelizer, 'VoiceProcessor'):
        print("🎯 Found VoiceProcessor in Rust library!")
        processor = rust_debabelizer.VoiceProcessor(stt_provider="soniox")
        print(f"✅ Created VoiceProcessor: {processor}")
    else:
        print("❌ No VoiceProcessor found in Rust library")
        
except ImportError as e:
    print(f"❌ Failed to import lib_internal: {e}")
    
    # Try alternative name
    try:
        import _internal as rust_debabelizer
        print("✅ Successfully imported _internal!")
        print(f"Available attributes: {dir(rust_debabelizer)}")
    except ImportError as e2:
        print(f"❌ Failed to import _internal: {e2}")
        
        print("\n🔍 Trying to understand the library structure...")
        # List .so files
        so_files = [f for f in os.listdir(target_path) if f.endswith('.so')]
        print(f"Available .so files: {so_files}")
        
        for so_file in so_files:
            print(f"\n📋 Attempting to load {so_file}...")
            so_path = os.path.join(target_path, so_file)
            try:
                # Try to inspect the library
                import importlib.util
                spec = importlib.util.spec_from_file_location("rust_lib", so_path)
                if spec:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print(f"✅ Successfully loaded {so_file}!")
                    print(f"Available attributes: {dir(module)}")
                    break
            except Exception as e:
                print(f"❌ Failed to load {so_file}: {e}")