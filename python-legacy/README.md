# Python Legacy Implementation

This directory contains the original Python implementation of Debabelizer, preserved for reference purposes.

## ⚠️ Important Notice

**This is the legacy Python implementation. The active development has moved to the Rust implementation in the `debabelizer/` directory.**

## Why Keep This?

- **Reference**: The Python implementation serves as a reference for the Rust port
- **API Design**: Shows the original API design and provider implementations
- **Testing**: The test suite provides comprehensive examples of expected behavior
- **Migration**: Helpful for understanding the migration path from Python to Rust

## Structure

```
python-legacy/
├── debabelizer/           # Main Python package
│   ├── core/             # Core processor and configuration
│   ├── providers/        # STT and TTS provider implementations
│   └── utils/           # Utility functions
└── debabelizer.egg-info/ # Package metadata
```

## DO NOT USE IN PRODUCTION

For all new development and production use, please use the Rust implementation. This Python code is kept solely for reference and should not be used in any active projects.