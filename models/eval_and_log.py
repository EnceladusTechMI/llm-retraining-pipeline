# models/eval_and_log.py

from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import mlflow
import os
import shutil
from datetime import datetime

# 🧩 Step 1: Load evaluation data from eval.json (predictions + ground_truth)
def load_eval_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 🧠 Step 2: Compute cosine similarity between each prediction and ground truth using embeddings
def compute_similarity(model, ground_truths, predictions):
    gt_embeddings = model.encode(ground_truths, convert_to_tensor=True)
    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    similarities = util.cos_sim(gt_embeddings, pred_embeddings)
    
    # Extract diagonal scores (prediction i vs. ground_truth i)
    scores = [float(similarities[i][i]) for i in range(len(similarities))]
    return scores

# ✅ Step 3: Evaluate model using average similarity score and threshold
def evaluate(eval_file_path, threshold=0.85):
    # Load pretrained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load eval data (list of dicts with question, prediction, ground_truth)
    eval_data = load_eval_data(eval_file_path)

    ground_truths = [item["ground_truth"] for item in eval_data]
    predictions = [item["prediction"] for item in eval_data]

    # Compute scores
    scores = compute_similarity(model, ground_truths, predictions)
    avg_score = np.mean(scores)

    passed = avg_score >= threshold

    # Print result summary
    print(f"\n📊 Average Semantic Similarity Score: {avg_score:.4f}")
    print("✅ Evaluation PASSED" if passed else "❌ Evaluation FAILED")

    return avg_score, passed

# 🪣 Step 4: Log evaluation results to MLflow
def log_to_mlflow(eval_file, score, passed, model_name="falcon-rw-1b", lora_path="models/lora_falcon_rw"):
    # Set MLflow to use local directory
    mlflow.set_tracking_uri("file:./mlruns")

    # Create unique run name with timestamp
    run_name = f"eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start MLflow logging
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("lora_adapter_path", lora_path)
        mlflow.log_metric("semantic_similarity_score", score)
        mlflow.set_tag("passed", str(passed))  # "True" or "False"
         # ✅ NEW: Set MLflow promotion tag
        stage = "production" if passed else "staging"
        mlflow.set_tag("stage", stage)

        # Upload eval.json for traceability
        mlflow.log_artifact(eval_file, artifact_path="evaluation_outputs")

        print(f"📦 MLflow run logged with ID: {run.info.run_id}")
        print(f"🏷️  Model tagged as '{stage}'")

# 🚀 Step 5: If model passed evaluation, promote it to the production folder
def promote_model_if_passed(source_dir, target_dir, passed):
    if passed:
        print(f"\n🚀 Promoting model from '{source_dir}' → '{target_dir}'...")

        # Clear old production model if exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Copy LoRA adapter files to production
        shutil.copytree(source_dir, target_dir)
        print("✅ Model promotion complete.")
    else:
        print("⚠️ Model not promoted — evaluation failed.")

# 🎯 Step 6: Run everything in sequence
if __name__ == "__main__":
    # Configuration paths
    eval_file = "eval_data/eval.json"
    threshold = 0.3  # Set lower threshold for testing
    lora_path = "models/lora_falcon_rw"  # Your fine-tuned model output folder
    production_path = "models/production/"  # Destination for promoted model
    model_name = "falcon-rw-1b"  # Model name for MLflow and tracking

    # Run evaluation
    score, passed = evaluate(eval_file, threshold)

    # Log to MLflow
    log_to_mlflow(eval_file, score, passed, model_name, lora_path)

    # If passed, promote the model
    promote_model_if_passed(lora_path, production_path, passed)
