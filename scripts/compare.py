from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "tiiuae/falcon-rw-1b"
prompt = "Explain the importance of renewable energy."

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompt, return_tensors="pt")

# 🧪 Base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
base_model.eval()
base_output = base_model.generate(**inputs, max_new_tokens=100)
print("\n🧪 Base Output:\n", tokenizer.decode(base_output[0], skip_special_tokens=True))

# ✅ Fine-tuned model
from peft import PeftModel
adapter_path = "models/lora_falcon_rw"
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
peft_model.eval()
peft_output = peft_model.generate(**inputs, max_new_tokens=100)
print("\n✅ Fine-Tuned Output:\n", tokenizer.decode(peft_output[0], skip_special_tokens=True))
