import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

def load_test_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_eval_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def run_inference(base_model_name, lora_dir, test_path, output_path):
    device = "cpu"  # (matching your training)

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.to(device)
    model.eval()

    test_data = load_test_data(test_path)
    results = []

    for item in tqdm(test_data):
        prompt = f"Q: {item['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated.replace(prompt, "").strip()

        # Fallback if generation fails
        if not generated:
            generated = "[NO ANSWER GENERATED]"

        results.append({
            "question": prompt,
            "ground_truth": item["ground_truth"],
            "prediction": generated
        })

    # ✅ Save even if no generation was good
    save_eval_data(results, output_path)
    print(f"✅ Saved {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="tiiuae/falcon-rw-1b")
    parser.add_argument("--lora_dir", default="models/lora_falcon_rw")
    parser.add_argument("--test_file", default="eval_data/test_set.json")
    parser.add_argument("--output_file", default="eval_data/eval.json")
    args = parser.parse_args()

    run_inference(
        args.base_model,
        args.lora_dir,
        args.test_file,
        args.output_file
    )
