from huggingface_hub import hf_hub_download
import os

# Set the target directory
target_dir = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models"

# Create directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Repository information
repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"

# Download the model file
# Note: GGUF repos typically contain multiple quantization versions
# You can specify which file to download, or download all files

try:
    print(f"Downloading model from {repo_id}...")
    print(f"Target directory: {target_dir}")
    
    # Option 1: Download a specific quantization (recommended)
    # Common options: Q4_K_M (good balance), Q5_K_M (better quality), Q8_0 (highest quality)
    filename = "Phi-3-mini-4k-instruct-q4.gguf"  # Adjust based on available files
    
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Successfully downloaded to: {file_path}")
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("\nTo see available files in the repository, run:")
    print(f"huggingface-cli scan-cache")
    print(f"\nOr visit: https://huggingface.co/{repo_id}/tree/main")