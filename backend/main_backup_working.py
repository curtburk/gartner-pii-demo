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

# GGUF Model paths - UPDATE THESE PATHS
BASE_MODEL_PATH = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/llama.cpp/models/qwen25_base_Q4_K_M.gguf"
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
        print("Loading GGUF models (this should be fast!)...")
        
        # Check if model files exist
        if not os.path.exists(BASE_MODEL_PATH):
            print(f"Base model not found at {BASE_MODEL_PATH}")
            return False
        if not os.path.exists(FINETUNED_MODEL_PATH):
            print(f"Finetuned model not found at {FINETUNED_MODEL_PATH}")
            return False
        
        print("Loading base model...")
        base_model = Llama(
            model_path=BASE_MODEL_PATH,
            n_gpu_layers=-1,  # Use GPU for all layers
            n_ctx=2048,       # Context window
            n_batch=512,      # Batch size for prompt processing
            verbose=False
        )
        print("Base model loaded!")
        
        print("Loading finetuned model...")
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
    """Create appropriate prompt for PII masking task - matching training format"""
    if is_finetuned:
        # Match EXACT format used in finetuning
        system = "You are a specialized PII/PHI detection and masking assistant. You identify and mask sensitive information in text while preserving the document's readability."
        instruction = f"Mask all personally identifiable information (PII) and protected health information (PHI) in the following document from the general domain:\n\n{text}"
        
        # Use Qwen format exactly as in training
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # More detailed prompt for base model
        system = "You are a specialized PII/PHI detection and masking assistant."
        instruction = f"""Mask all personally identifiable information in the text below.

Rules:
- Replace person names with [NAME]
- Replace phone numbers with [PHONE]
- Replace email addresses with [EMAIL]  
- Replace social security numbers with [SSN]
- Replace physical addresses with [ADDRESS]
- Replace dates with [DATE]
- Replace credit card numbers with [CREDIT_CARD]
- Replace ID numbers with [ID]

Text: {text}"""
        
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt

def generate_with_gguf(model, prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> tuple:
    """Generate masked text using GGUF model"""
    start_time = time.time()
    
    try:
        response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "\n\n", "User:", "Human:", "Assistant:"],
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
        "status": "PII Masking Demo (GGUF) Running",
        "models_loaded": models_loaded,
        "base_model": "Qwen2.5-32B Q4_K_M" if models_loaded else "Not loaded",
        "finetuned_model": "PII Detector Q4_K_M" if models_loaded else "Not loaded"
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
        # Generate with base model
        base_prompt = create_pii_prompt(request.text, is_finetuned=False)
        base_output, base_time = generate_with_gguf(
            base_model, 
            base_prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Generate with finetuned model
        finetuned_prompt = create_pii_prompt(request.text, is_finetuned=True)
        finetuned_output, finetuned_time = generate_with_gguf(
            finetuned_model,
            finetuned_prompt,
            request.max_tokens,
            request.temperature
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

@app.post("/mask_pii_sequential")
async def mask_pii_sequential(request: PIIRequest):
    """Process with one model at a time to save memory"""
    global base_model, finetuned_model
    
    try:
        # Load and process with base model
        print("Loading base model for processing...")
        if base_model is None:
            base_model = Llama(
                model_path=BASE_MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=2048,
                n_batch=512,
                verbose=False
            )
        
        base_prompt = create_pii_prompt(request.text, is_finetuned=False)
        base_output, base_time = generate_with_gguf(
            base_model, 
            base_prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Free base model memory
        del base_model
        base_model = None
        gc.collect()
        
        # Load and process with finetuned model
        print("Loading finetuned model for processing...")
        finetuned_model = Llama(
            model_path=FINETUNED_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            verbose=False
        )
        
        finetuned_prompt = create_pii_prompt(request.text, is_finetuned=True)
        finetuned_output, finetuned_time = generate_with_gguf(
            finetuned_model,
            finetuned_prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Free finetuned model memory
        del finetuned_model
        finetuned_model = None
        gc.collect()
        
        return PIIResponse(
            original_text=request.text,
            base_model_output=base_output,
            finetuned_model_output=finetuned_output,
            base_model_time=round(base_time, 2),
            finetuned_model_time=round(finetuned_time, 2),
            timestamp=datetime.now().isoformat(),
            status="sequential"
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
    print("Loading models on startup...")
    success = load_models()
    if success:
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Failed to load models on startup")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
