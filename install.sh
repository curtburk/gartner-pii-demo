#!/bin/bash

echo "======================================"
echo "PII Masking Demo Installer"
echo "======================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install with: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

echo "✔ Python 3 found: $(python3 --version)"

# Use existing virtual environment
echo ""
echo "Using existing virtual environment: new-ft-env"
if [ ! -d "new-ft-env" ]; then
    echo "❌ Virtual environment 'new-ft-env' not found!"
    echo "Please ensure 'new-ft-env' exists in the current directory"
    exit 1
else
    echo "✔ Found virtual environment 'new-ft-env'"
fi

# Activate virtual environment
source new-ft-env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130


# ============================================================================
# CUDA 13 / GB10 Blackwell Configuration
# ============================================================================
echo ""
echo "======================================"
echo "🔧 Configuring for GB10 Blackwell GPU"
echo "======================================"

# Check for CUDA 13
CUDA13_NVCC="/usr/local/cuda-13.0/bin/nvcc"
if [ -f "$CUDA13_NVCC" ]; then
    echo "✓ Found CUDA 13 compiler: $CUDA13_NVCC"
    
    echo ""
    echo "Installing llama-cpp-python with CUDA 13 support..."
    echo "This may take several minutes to compile..."
    echo ""
    
    # Build llama-cpp-python with correct CUDA 13 compiler for Blackwell
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA13_NVCC -DCMAKE_CUDA_ARCHITECTURES=120" \
        pip install llama-cpp-python --no-cache-dir --force-reinstall
    
    if [ $? -eq 0 ]; then
        echo "✓ llama-cpp-python installed with CUDA 13 / Blackwell support"
    else
        echo "❌ Failed to build llama-cpp-python with CUDA support"
        echo "Falling back to CPU-only version..."
        pip install llama-cpp-python
    fi
else
    echo "⚠️  CUDA 13 not found at $CUDA13_NVCC"
    echo "Checking for other CUDA installations..."
    
    # Try to find any CUDA installation
    if [ -d "/usr/local/cuda" ]; then
        CUDA_NVCC="/usr/local/cuda/bin/nvcc"
        if [ -f "$CUDA_NVCC" ]; then
            CUDA_VERSION=$($CUDA_NVCC --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            echo "Found CUDA $CUDA_VERSION at $CUDA_NVCC"
            
            CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA_NVCC" \
                pip install llama-cpp-python --no-cache-dir --force-reinstall
        fi
    else
        echo "No CUDA found. Installing CPU-only version..."
        pip install llama-cpp-python
    fi
fi

# Verify llama-cpp-python installation
echo ""
echo "Verifying llama-cpp-python installation..."
if python3 -c "from llama_cpp import Llama; print('✓ llama-cpp-python OK')" 2>/dev/null; then
    echo "✓ llama-cpp-python installed successfully"
else
    echo "❌ llama-cpp-python installation failed"
    exit 1
fi
# ============================================================================
# ============================================================================

# Install backend dependencies
echo ""
echo "Installing dependencies..."
cd backend
pip install -r requirements.txt

echo ""
echo "======================================"
echo "⚠️  Model Setup Information"
echo "======================================"
echo ""
echo "This demo uses two large language models:"
echo "1. Base Model: Qwen/Qwen2.5-32B-Instruct"
echo "2. Finetuned Model: pii_detector_Qwen32B_FTmerged"
echo ""
echo "IMPORTANT:"
echo "- Models will be loaded when you click 'Load Models' in the web interface"
echo "- Each model is ~32GB, so loading may take several minutes"
echo "- Ensure you have sufficient GPU memory (recommended: 40GB+ VRAM)"
echo "- Models will use 8-bit quantization to reduce memory requirements"
echo ""
echo "If your finetuned model is in a local directory, update the path in:"
echo "  backend/main.py -> FINETUNED_MODEL_NAME variable"
echo ""

cd ..

# Create offline_responses.json for fallback
echo ""
echo "Creating offline response database..."
cd backend
cat > offline_responses.json <<'EOF'
{
    "name": "Names should be replaced with [NAME] to protect identity",
    "phone": "Phone numbers should be replaced with [PHONE]",
    "email": "Email addresses should be replaced with [EMAIL]",
    "ssn": "Social Security Numbers should be replaced with [SSN]",
    "address": "Physical addresses should be replaced with [ADDRESS]",
    "date": "Dates of birth should be replaced with [DATE]",
    "credit": "Credit card numbers should be replaced with [CREDIT_CARD]",
    "id": "ID numbers should be replaced with [ID]",
    "default": "This text contains PII that should be masked for privacy protection"
}
EOF
echo "✔ Offline response database created"

cd ..

echo ""
echo "======================================"
echo "✅ Installation Complete!"
echo "======================================"
echo ""
echo "To start the demo:"
echo "  ./start_demo_remote.sh"
echo ""
echo "Then access from your Windows laptop:"
echo "  http://YOUR_SERVER_IP:8080"
echo ""
echo "Note: Update the IP address in start_demo_remote.sh to match your server"
echo ""
