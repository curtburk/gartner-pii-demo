#!/bin/bash

# Update the subtitle
sed -i 's/Base Qwen2\.5-32B vs Finetuned PII Detector/TinyLlama-1.1B vs Finetuned Qwen2.5-32B/g' index.html
sed -i 's/Gemma2-9B vs Finetuned Qwen2\.5-32B PII Detector/TinyLlama-1.1B vs Finetuned Qwen2.5-32B/g' index.html

# Update base model badge
sed -i 's/Gemma2-9B-IT/TinyLlama-1.1B/g' index.html
sed -i 's/Qwen2\.5-32B-Instruct/TinyLlama-1.1B/g' index.html

# Update finetuned model badge  
sed -i 's/PII Detector Qwen32B/Finetuned Qwen2.5-32B/g' index.html
sed -i 's/Finetuned Qwen2\.5-32B/Qwen2.5-32B Finetuned/g' index.html

# Update any references to model sizes
sed -i 's/9B/1.1B/g' index.html

