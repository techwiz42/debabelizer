# PyPI Upload Instructions

## Prerequisites
1. **PyPI Account**: Create account at https://pypi.org/account/register/
2. **API Token**: Generate API token at https://pypi.org/manage/account/token/
3. **Install twine**: `pip install twine`

## Upload Steps

### 1. Test Upload (Optional - uploads to TestPyPI first)
```bash
# Upload to TestPyPI first for testing
twine upload --repository testpypi target/wheels/debabelizer-0.1.4-cp38-abi3-manylinux_2_34_x86_64.whl
```

### 2. Production Upload to PyPI
```bash
# Upload to real PyPI
twine upload target/wheels/debabelizer-0.1.4-cp38-abi3-manylinux_2_34_x86_64.whl
```

### 3. Alternative: Use maturin directly
```bash
# Set up credentials first
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here

# Upload using maturin
maturin publish --release
```

## Package Details
- **Package Name**: debabelizer
- **Version**: 0.1.4
- **Wheel Built**: `/home/peter/debabelizer/target/wheels/debabelizer-0.1.4-cp38-abi3-manylinux_2_34_x86_64.whl`
- **Compatible with**: Python 3.8+ on Linux x86_64

## Important Notes
1. **Package Name Availability**: Make sure "debabelizer" is available on PyPI
2. **Version Management**: You can only upload each version once
3. **API Token Security**: Keep your PyPI API token secure and never commit it to git

## After Upload
Users will be able to install with:
```bash
pip install debabelizer
```

## Build Additional Wheels (Optional)
For broader compatibility, you might want to build wheels for other platforms:
```bash
# Build for different architectures/platforms
maturin build --release --target x86_64-apple-darwin     # macOS Intel
maturin build --release --target aarch64-apple-darwin    # macOS Apple Silicon  
maturin build --release --target x86_64-pc-windows-msvc  # Windows
```