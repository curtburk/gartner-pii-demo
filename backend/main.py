from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
from llama_cpp import Llama
import re
import time
import os
import gc

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GGUF Model paths - TinyLlama vs Finetuned Qwen2.5-32B
BASE_MODEL_PATH = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FINETUNED_MODEL_PATH = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models/pii_detector_Q4_K_M.gguf"

# Global variables for models
base_model = None
finetuned_model = None
models_loaded = False

class PIIRequest(BaseModel):
    text: str
    max_tokens: int = 256
    temperature: float = 0.1

class PIIResponse(BaseModel):
    original_text: str
    base_model_output: str
    finetuned_model_output: str
    base_model_time: float
    finetuned_model_time: float
    timestamp: str
    status: str

def load_models():
    """Load both GGUF models into memory"""
    global base_model, finetuned_model, models_loaded
    
    try:
        print("Loading GGUF models...")
        
        # Check if model files exist
        if not os.path.exists(BASE_MODEL_PATH):
            print(f"Base model (TinyLlama) not found at {BASE_MODEL_PATH}")
            return False
        if not os.path.exists(FINETUNED_MODEL_PATH):
            print(f"Finetuned model not found at {FINETUNED_MODEL_PATH}")
            return False
        
        print("Loading TinyLllam base model...")
        base_model = Llama(
            model_path=BASE_MODEL_PATH,
            n_gpu_layers=-1,  # Use GPU for all layers
            n_ctx=2048,       # Context window
            n_batch=512,      # Batch size for prompt processing
            verbose=False
        )
        print("TinyLlama loaded!")
        
        print("Loading finetuned Qwen2.5-32B model...")
        finetuned_model = Llama(
            model_path=FINETUNED_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            verbose=False
        )
        print("Finetuned model loaded!")
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def create_pii_prompt(text: str, is_finetuned: bool = False) -> str:
    """Create appropriate prompt for PII masking task"""
    if is_finetuned:
        # Finetuned Qwen model - use exact training format
        system = "You are a specialized PII/PHI detection and masking assistant. You identify and mask sensitive information in text while preserving the document's readability."
        instruction = f"Mask all personally identifiable information (PII) and protected health information (PHI) in the following document from the general domain:\n\n{text}"
        
        # Qwen format
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # TinyLlama base model - use TinyLlama format
        instruction = f"Identify and replace all personally identifiable information (PII) in this text with [MASKED] labels:\n\n{text}"
        
        # TinyLlama format
        prompt = f"<|system|>\nReplace personal information with [MASKED]</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
    
    return prompt

def generate_with_gguf(model, prompt: str, max_tokens: int = 256, temperature: float = 0.1, is_gemma: bool = False) -> tuple:
    """Generate masked text using GGUF model"""
    start_time = time.time()
    
    try:
        # Different stop tokens for different models
        if is_gemma:
            stop_tokens = ["<end_of_turn>", "<start_of_turn>", "\n\n"]
        else:
            stop_tokens = ["<|im_end|>", "<|im_start|>", "\n\n"]
        
        response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_tokens,
            echo=False  # Don't include prompt in response
        )
        
        output = response['choices'][0]['text'].strip()
        elapsed_time = time.time() - start_time
        
        return output, elapsed_time
        
    except Exception as e:
        print(f"Error during generation: {e}")
        elapsed_time = time.time() - start_time
        return f"Error: {str(e)}", elapsed_time

@app.get("/")
def read_root():
    return {
        "status": "PII Masking Demo Running",
        "models_loaded": models_loaded,
        "base_model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" if models_loaded else "Not loaded",
        "finetuned_model": "Qwen2.5-32B Finetuned Q4_K_M" if models_loaded else "Not loaded"
    }

@app.get("/load_models")
async def load_models_endpoint():
    """Endpoint to trigger model loading"""
    if models_loaded:
        return {"status": "Models already loaded"}
    
    success = load_models()
    if success:
        return {"status": "Models loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load models")

@app.post("/mask_pii")
async def mask_pii(request: PIIRequest):
    """Process text through both models and return masked versions"""
    
    if not models_loaded:
        # Try to load models automatically
        success = load_models()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Models not loaded. Please check model paths."
            )
    
    try:
        # Generate with TinyLlama base model
        base_prompt = create_pii_prompt(request.text, is_finetuned=False)
        print(f"DEBUG: Base prompt: {base_prompt[:100]}")
        print(f"DEBUG: Using base_model: {type(base_model)}")
        base_output, base_time = generate_with_gguf(
            base_model, 
            base_prompt,
            request.max_tokens,
            request.temperature,
            is_gemma=True
        )
        
        # Generate with finetuned Qwen model
        finetuned_prompt = create_pii_prompt(request.text, is_finetuned=True)
        print(f"DEBUG: FT prompt: {finetuned_prompt[:100]}")
        print(f"DEBUG: Using finetuned_model: {type(finetuned_model)}")
        finetuned_output, finetuned_time = generate_with_gguf(
            finetuned_model,
            finetuned_prompt,
            request.max_tokens,
            request.temperature,
            is_gemma=False
        )
        
        return PIIResponse(
            original_text=request.text,
            base_model_output=base_output,
            finetuned_model_output=finetuned_output,
            base_model_time=round(base_time, 2),
            finetuned_model_time=round(finetuned_time, 2),
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_offline")
async def test_offline(request: PIIRequest):
    """Test endpoint without models"""
    text = request.text
    
    # Simple regex-based masking
    import re
    output = text
    output = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', output)
    output = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', output)
    output = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', output)
    output = re.sub(r'\b\d{5}\b', '[ZIP]', output)
    
    return PIIResponse(
        original_text=text,
        base_model_output=output,
        finetuned_model_output=output,
        base_model_time=0.1,
        finetuned_model_time=0.1,
        timestamp=datetime.now().isoformat(),
        status="test"
    )

if __name__ == "__main__":
    # Load models on startup
    print("="*60)
    print("PII Masking Demo - Model Comparison")
    print("Base Model: TinyLlama (1.1 billion parameters)")
    print("Finetuned Model: Qwen2.5-32B PII Detector (32 billion parameters)")
    print("="*60)
    
    print("\nLoading models on startup...")
    success = load_models()
    if success:
        print("✅ Both models loaded successfully!")
    else:
        print("⚠️ Failed to load models on startup")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
