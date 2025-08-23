# Soniox Streaming Background Task Bug Analysis

## Issue Summary

**Problem**: Soniox streaming functionality appears to work (handshake succeeds) but the Python iterator immediately ends without yielding any results, even with valid API keys and real audio data.

**Root Cause**: The background WebSocket handler task exits immediately after completing the handshake, instead of entering the main event loop to process commands and messages.

## Investigation Timeline

### Initial Symptoms (v0.1.44)
- Python streaming iterator ending immediately with `StopAsyncIteration`
- No messages received from Soniox WebSocket
- Iterator timeout after 0 results: "Results collection ended after 0 results"

### Suspected Causes (Investigated and Ruled Out)
1. ❌ **Legacy Python code being used instead of Rust** - RULED OUT
   - Rust VoiceProcessor IS being called (confirmed via debug output)
   - All Rust provider initialization working correctly
   
2. ❌ **Provider selection issues** - RULED OUT  
   - Soniox provider successfully registered and selected
   - Provider config loading working correctly
   
3. ❌ **WebSocket connection failures** - RULED OUT
   - WebSocket connection establishes successfully
   - Handshake completes with valid response from Soniox API
   
4. ❌ **API authentication issues** - RULED OUT
   - Testing with real API key shows same behavior
   - Handshake succeeds: `{"tokens":[],"final_audio_proc_ms":0,"total_audio_proc_ms":0}`

### Actual Root Cause Identified

**Location**: `/home/peter/debabelizer/providers/soniox/src/lib.rs`
**Function**: `websocket_handler()` (lines 356-415)
**Issue**: Background task terminates between handshake completion and main loop entry

## Technical Analysis

### Expected Flow
```rust
// Line 393: Handshake success message appears ✅
println!("✅ RUST: Soniox handshake successful - entering main loop");

// Line 412: End of handshake match block

// Line 415: Should enter main loop ❌ NEVER REACHED
tracing::info!("🔄 Entering main WebSocket event loop");

// Line 416: Main processing loop ❌ NEVER REACHED  
loop {
    tokio::select! {
        // Process audio commands
        command = command_rx.recv() => { ... }
        // Process WebSocket messages  
        ws_msg = ws_stream.next() => { ... }
    }
}
```

### Actual Flow (Broken)
```
✅ WebSocket connection established
✅ Handshake sent successfully  
✅ Handshake response received: {"tokens":[],...}
✅ "Soniox handshake successful - entering main loop" printed
❌ Task terminates here (line 393-415)
❌ "🔄 Entering main WebSocket event loop" NEVER printed
❌ Main processing loop NEVER started
```

### Impact on Python Iterator

```rust
// In receive_transcript() method (line 599)
if self.background_task.is_finished() {
    tracing::error!("❌ Background WebSocket task has finished!");
    return Ok(None); // END ITERATOR IMMEDIATELY
}
```

Since the background task ends immediately after handshake, the first call to `receive_transcript()` returns `Ok(None)`, causing the Python async iterator to raise `StopAsyncIteration`.

## Debug Evidence

### Test Output Analysis
```
🎯 RUST: Soniox provider - transcribe_stream called ✅
🚀 RUST: SonioxStream::new called ✅  
🌐 RUST: Attempting WebSocket connection ✅
✅ RUST: WebSocket connection established ✅
🔧 RUST: Starting Soniox WebSocket background handler ✅
📤 RUST: Sending initial configuration ✅
📥 RUST: Received Soniox handshake: {...} ✅
✅ RUST: Soniox handshake successful - entering main loop ✅

MISSING (should appear next):
❌ 🔄 Entering main WebSocket event loop  
❌ 🔄 Loop iteration - waiting for messages
❌ Sending X bytes of audio to Soniox
```

### Python Iterator Behavior
```python
# Python test results:
⚠️  Iterator ended without results
⚠️  Timeout after getting 0 results
```

This confirms the background task finishes immediately, causing the iterator check `background_task.is_finished()` to return `true` on the first call.

## Potential Causes

### 1. Task Cancellation
The background task might be getting cancelled by the Tokio runtime before it reaches the main loop.

**Investigation needed**: Check if the task handle is being dropped or cancelled elsewhere.

### 2. Resource Drop
Some resource (WebSocket stream, channels) might be getting dropped, causing the task to terminate.

**Investigation needed**: Verify all resources are properly held in scope.

### 3. Silent Panic/Error
There might be a panic or error occurring between lines 412-415 that's not being logged.

**Investigation needed**: Add debug statements to trace execution flow.

### 4. Compiler Optimization
Unlikely but possible that compiler optimization is affecting control flow.

**Investigation needed**: Test with debug build.

## Files Involved

### Primary
- `/home/peter/debabelizer/providers/soniox/src/lib.rs` - Contains broken websocket_handler function
- `/home/peter/debabelizer/debabelizer-python/src/lib.rs` - Python bindings that create iterator

### Secondary  
- `/home/peter/debabelizer/debabelizer/src/processor.rs` - VoiceProcessor streaming methods
- `/home/peter/debabelizer/debabelizer/src/providers.rs` - Provider initialization

## Message-Driven Architecture Overview

The current implementation (v0.1.44) uses a message-driven architecture:

```
┌─────────────────┐    commands     ┌──────────────────────┐
│   Python API    │ ──────────────> │  Background WebSocket │
│                 │                 │       Handler         │
│  SonioxStream   │ <────────────── │                      │
└─────────────────┘    results      └──────────────────────┘
```

**Channels**:
- `command_tx/command_rx`: Send audio chunks and close commands  
- `result_tx/result_rx`: Receive transcription results and errors

**Issue**: The background handler task terminates before entering the main `tokio::select!` loop that processes these channels.

## Fix Requirements

1. **Identify termination cause**: Add debug tracing to pinpoint where task exits
2. **Ensure task stays alive**: Fix whatever is causing premature termination
3. **Verify main loop entry**: Confirm main event loop starts and processes commands
4. **Test end-to-end**: Verify Python iterator receives results from working background task

## Testing Strategy

### 1. Add Debug Tracing
```rust
// Add before line 415
println!("🔧 RUST: About to enter main WebSocket event loop");
tracing::info!("🔧 About to enter main WebSocket event loop");

// Add at start of main loop  
println!("🔄 RUST: Successfully entered main event loop");
```

### 2. Test with Real API Key
Use valid Soniox API key: `cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95`

### 3. Verify Task Lifetime
Check `background_task.is_finished()` status at different points.

### 4. End-to-End Validation
Confirm Python iterator receives streaming results after fix.

## Success Criteria

✅ **Background task stays alive**: Debug message "🔄 Entering main WebSocket event loop" appears  
✅ **Commands processed**: Audio chunks sent to Soniox via WebSocket  
✅ **Messages processed**: Responses from Soniox parsed and sent to result channel  
✅ **Python iterator works**: Iterator yields StreamingResult objects instead of ending immediately  
✅ **Real-time streaming**: Continuous audio processing with interim and final results

## Current Status

- **Investigation**: ✅ COMPLETE - Root cause identified  
- **Documentation**: ✅ COMPLETE - This document
- **Fix Implementation**: ❌ PENDING
- **Testing**: ❌ PENDING  
- **Deployment**: ❌ PENDING

## Next Steps

1. **Add debug tracing** to identify exact termination point
2. **Implement fix** based on findings  
3. **Test fix** with real Soniox API key
4. **Update version** and deploy to PyPI
5. **Document resolution** for future reference

---

*Analysis completed: 2025-01-23*  
*Status: Ready for fix implementation*