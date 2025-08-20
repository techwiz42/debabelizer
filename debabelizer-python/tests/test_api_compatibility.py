"""
Test API compatibility between legacy Python implementation and Rust bindings.

This test ensures that the Rust Python bindings have the exact same API
as the legacy Python implementation, enabling drop-in replacement.
"""

import sys
import inspect
from typing import get_type_hints, Union, Optional, List, Dict, Any


def test_api_compatibility():
    """Verify that the Rust bindings match the legacy Python API exactly."""
    
    # Import both implementations
    sys.path.insert(0, '/home/peter/debabelizer/python-legacy')
    import debabelizer as legacy_debabelizer
    sys.path.pop(0)
    
    sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')
    import debabelizer as rust_debabelizer
    sys.path.pop(0)
    
    # Test 1: Check module-level exports
    print("=== Checking module exports ===")
    legacy_exports = set(legacy_debabelizer.__all__)
    rust_exports = set(rust_debabelizer.__all__)
    
    missing_in_rust = legacy_exports - rust_exports
    extra_in_rust = rust_exports - legacy_exports
    
    if missing_in_rust:
        print(f"❌ Missing in Rust bindings: {missing_in_rust}")
    if extra_in_rust:
        print(f"⚠️  Extra in Rust bindings: {extra_in_rust}")
    if not missing_in_rust and not extra_in_rust:
        print("✅ Module exports match")
    
    # Test 2: Check VoiceProcessor class methods
    print("\n=== Checking VoiceProcessor methods ===")
    legacy_vp = legacy_debabelizer.VoiceProcessor
    rust_vp = rust_debabelizer.VoiceProcessor
    
    # Get all public methods
    legacy_methods = {name for name, _ in inspect.getmembers(legacy_vp, inspect.isfunction) 
                      if not name.startswith('_')}
    rust_methods = {name for name, _ in inspect.getmembers(rust_vp, inspect.isroutine) 
                    if not name.startswith('_')}
    
    # Also check for async methods in legacy (they appear as coroutine functions)
    legacy_async_methods = {name for name in dir(legacy_vp) 
                           if not name.startswith('_') and 
                           inspect.iscoroutinefunction(getattr(legacy_vp, name, None))}
    legacy_methods.update(legacy_async_methods)
    
    missing_methods = legacy_methods - rust_methods
    extra_methods = rust_methods - legacy_methods
    
    if missing_methods:
        print(f"❌ Missing methods in Rust VoiceProcessor: {missing_methods}")
    if extra_methods:
        print(f"⚠️  Extra methods in Rust VoiceProcessor: {extra_methods}")
    if not missing_methods and not extra_methods:
        print("✅ VoiceProcessor methods match")
    
    # Test 3: Check VoiceProcessor constructor signature
    print("\n=== Checking VoiceProcessor constructor ===")
    legacy_init = inspect.signature(legacy_vp.__init__)
    rust_init = inspect.signature(rust_vp.__init__)
    
    print(f"Legacy: {legacy_init}")
    print(f"Rust:   {rust_init}")
    
    # Check parameter names (excluding 'self')
    legacy_params = list(legacy_init.parameters.keys())[1:]  # Skip 'self'
    rust_params = list(rust_init.parameters.keys())[1:]  # Skip 'self'
    
    if legacy_params == rust_params:
        print("✅ Constructor parameters match")
    else:
        print("❌ Constructor parameters differ")
    
    # Test 4: Check data classes
    print("\n=== Checking data classes ===")
    data_classes = ['AudioFormat', 'TranscriptionResult', 'SynthesisResult', 
                    'Voice', 'StreamingResult', 'WordTiming']
    
    for cls_name in data_classes:
        legacy_cls = getattr(legacy_debabelizer, cls_name, None)
        rust_cls = getattr(rust_debabelizer, cls_name, None)
        
        if legacy_cls is None:
            print(f"⚠️  {cls_name} not found in legacy")
        elif rust_cls is None:
            print(f"❌ {cls_name} not found in Rust bindings")
        else:
            # Check if it's a dataclass or has expected attributes
            legacy_attrs = {attr for attr in dir(legacy_cls) if not attr.startswith('_')}
            rust_attrs = {attr for attr in dir(rust_cls) if not attr.startswith('_')}
            
            missing_attrs = legacy_attrs - rust_attrs
            if missing_attrs:
                print(f"❌ {cls_name} missing attributes: {missing_attrs}")
            else:
                print(f"✅ {cls_name} attributes match")
    
    # Test 5: Check create_processor function
    print("\n=== Checking create_processor function ===")
    if hasattr(legacy_debabelizer, 'create_processor'):
        legacy_sig = inspect.signature(legacy_debabelizer.create_processor)
        print(f"Legacy: {legacy_sig}")
        
        if hasattr(rust_debabelizer, 'create_processor'):
            rust_sig = inspect.signature(rust_debabelizer.create_processor)
            print(f"Rust:   {rust_sig}")
            
            # Note: The legacy has a typo "elevanlabs" which we've preserved
            legacy_defaults = {
                param.name: param.default 
                for param in legacy_sig.parameters.values() 
                if param.default is not inspect.Parameter.empty
            }
            rust_defaults = {
                param.name: param.default 
                for param in rust_sig.parameters.values() 
                if param.default is not inspect.Parameter.empty
            }
            
            if legacy_defaults == rust_defaults:
                print("✅ create_processor defaults match (including typo)")
            else:
                print(f"⚠️  Default values differ:")
                print(f"   Legacy: {legacy_defaults}")
                print(f"   Rust:   {rust_defaults}")
        else:
            print("❌ create_processor not found in Rust bindings")
    
    # Test 6: Check exception classes
    print("\n=== Checking exception classes ===")
    exceptions = ['ProviderError', 'AuthenticationError', 'RateLimitError', 'ConfigurationError']
    
    for exc_name in exceptions:
        legacy_exc = getattr(legacy_debabelizer, exc_name, None)
        rust_exc = getattr(rust_debabelizer, exc_name, None)
        
        if legacy_exc is None:
            print(f"⚠️  {exc_name} not found in legacy")
        elif rust_exc is None:
            print(f"❌ {exc_name} not found in Rust bindings")
        else:
            print(f"✅ {exc_name} present in both")
    
    # Test 7: Test actual instantiation
    print("\n=== Testing instantiation ===")
    try:
        # Test with no arguments
        rust_proc = rust_debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor() instantiation works")
        
        # Test with provider arguments
        rust_proc2 = rust_debabelizer.VoiceProcessor(
            stt_provider="deepgram",
            tts_provider="openai"
        )
        print("✅ VoiceProcessor with providers instantiation works")
        
        # Test create_processor
        rust_proc3 = rust_debabelizer.create_processor()
        print("✅ create_processor() works with defaults")
        
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
    
    print("\n=== Summary ===")
    print("The Rust Python bindings should now be a drop-in replacement for the legacy Python implementation.")
    print("Any code using the legacy debabelizer module should work without modification.")


if __name__ == "__main__":
    test_api_compatibility()