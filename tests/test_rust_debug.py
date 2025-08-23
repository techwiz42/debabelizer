#!/usr/bin/env python3

import sys
import os

# Try to add target directory where the .so file might be
sys.path.insert(0, "/home/peter/debabelizer/target/release")
sys.path.insert(0, "/home/peter/debabelizer/debabelizer-python/target/release")

# Try to find the compiled library
search_paths = [
    "/home/peter/debabelizer/target/release/_internal.so",
    "/home/peter/debabelizer/debabelizer-python/target/release/_internal.so",
    "/home/peter/debabelizer/target/release/_internal.cpython*.so",
    "/home/peter/debabelizer/debabelizer-python/target/release/_internal.cpython*.so"
]

for path in search_paths:
    if '*' in path:
        import glob
        matches = glob.glob(path)
        if matches:
            print(f"Found compiled library: {matches[0]}")
            break
    elif os.path.exists(path):
        print(f"Found compiled library: {path}")
        break
else:
    print("❌ No compiled library found")
    print("Available files in target/release:")
    target_dirs = [
        "/home/peter/debabelizer/target/release",
        "/home/peter/debabelizer/debabelizer-python/target/release"
    ]
    for target_dir in target_dirs:
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            print(f"  {target_dir}: {files}")

print("🔍 Attempting to run a simple test to see if we get debug output...")

# Just try to run our existing test
print("Running existing test to see if we get debug messages...")
os.system("python3 /home/peter/debabelizer/test_message_driven_debug.py")