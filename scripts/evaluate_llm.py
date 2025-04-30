# scripts/evaluate_llm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "tiiuae/falcon-rw-1b"
adapter_path = "models/lora_falcon_rw"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Test prompt
prompt = "Explain the importance of renewable energy."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

print("\n🧪 Prompt:", prompt)
print("📘 Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
