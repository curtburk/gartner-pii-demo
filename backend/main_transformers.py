from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import time
import traceback

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # Will use HF cache if already downloaded
FINETUNED_MODEL_NAME = "/home/curtburk/Desktop/Demo-projects/Fine-tuning-demo/pii_detector_Qwen32B_FTmerged"

# Global variables for models
base_model = None
base_tokenizer = None
finetuned_model = None
finetuned_tokenizer = None
models_loaded = False

class PIIRequest(BaseModel):
    text: str
    max_length: int = 512
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
    """Load both models into memory"""
    global base_model, base_tokenizer, finetuned_model, finetuned_tokenizer, models_loaded
    
    try:
        print("Loading base model...")
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Use 8-bit quantization to save memory
        )
        print("Base model loaded successfully!")
        
        print("Loading finetuned model...")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_NAME)
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # Allow CPU offloading
        )
        print("Finetuned model loaded successfully!")
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

def create_pii_prompt(text: str, is_finetuned: bool = False) -> str:
    """Create appropriate prompt for PII masking task"""
    if is_finetuned:
        # Assuming the finetuned model was trained with a specific format
        prompt = f"""Task: Mask all personally identifiable information (PII) in the following text.
Replace names with [NAME], phone numbers with [PHONE], email addresses with [EMAIL], 
social security numbers with [SSN], addresses with [ADDRESS], and dates with [DATE].

Text: {text}

Masked Text:"""
    else:
        # More detailed prompt for base model
        prompt = f"""You are a PII (Personally Identifiable Information) detection and masking assistant.
Your task is to identify and mask all PII in the given text.

Rules:
- Replace person names with [NAME]
- Replace phone numbers with [PHONE]
- Replace email addresses with [EMAIL]
- Replace social security numbers with [SSN]
- Replace physical addresses with [ADDRESS]
- Replace birth dates with [DATE]
- Replace credit card numbers with [CREDIT_CARD]
- Replace ID numbers with [ID]

Original text: {text}

Please provide the text with all PII masked:"""
    
    return prompt

def generate_masked_text(model, tokenizer, prompt: str, max_length: int = 512, temperature: float = 0.1) -> tuple:
    """Generate masked text using the model"""
    start_time = time.time()
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=False if temperature < 0.1 else True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract just the masked text part
        if "Masked Text:" in full_output:
            masked_text = full_output.split("Masked Text:")[-1].strip()
        elif "Please provide the text with all PII masked:" in full_output:
            masked_text = full_output.split("Please provide the text with all PII masked:")[-1].strip()
        else:
            # Fallback: take everything after the original text
            masked_text = full_output.split(prompt)[-1].strip()
        
        elapsed_time = time.time() - start_time
        
        return masked_text, elapsed_time
        
    except Exception as e:
        print(f"Error during generation: {e}")
        elapsed_time = time.time() - start_time
        return f"Error: {str(e)}", elapsed_time

@app.get("/")
def read_root():
    return {
        "status": "PII Masking Demo API Running",
        "models_loaded": models_loaded,
        "base_model": BASE_MODEL_NAME if models_loaded else "Not loaded",
        "finetuned_model": FINETUNED_MODEL_NAME if models_loaded else "Not loaded"
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
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please call /load_models first"
        )
    
    try:
        # Generate with base model
        base_prompt = create_pii_prompt(request.text, is_finetuned=False)
        base_output, base_time = generate_masked_text(
            base_model, 
            base_tokenizer, 
            base_prompt,
            request.max_length,
            request.temperature
        )
        
        # Generate with finetuned model
        finetuned_prompt = create_pii_prompt(request.text, is_finetuned=True)
        finetuned_output, finetuned_time = generate_masked_text(
            finetuned_model,
            finetuned_tokenizer,
            finetuned_prompt,
            request.max_length,
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_offline")
async def test_offline(request: PIIRequest):
    """Test endpoint that works without models loaded"""
    
    # Simple regex-based PII masking for testing
    text = request.text
    
    # Base model simulation (less accurate)
    base_output = text
    base_output = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', base_output)
    base_output = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', base_output)
    base_output = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', base_output)
    
    # Finetuned model simulation (more accurate)
    finetuned_output = text
    finetuned_output = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', finetuned_output)
    finetuned_output = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', finetuned_output)
    finetuned_output = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', finetuned_output)
    finetuned_output = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', finetuned_output)
    finetuned_output = re.sub(r'\b\d{1,5}\s+[\w\s]+\b(?=,|\.|$)', '[ADDRESS]', finetuned_output)
    
    return PIIResponse(
        original_text=request.text,
        base_model_output=base_output,
        finetuned_model_output=finetuned_output,
        base_model_time=0.5,
        finetuned_model_time=0.3,
        timestamp=datetime.now().isoformat(),
        status="test_mode"
    )

if __name__ == "__main__":
    # Optionally load models on startup
    # load_models()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)