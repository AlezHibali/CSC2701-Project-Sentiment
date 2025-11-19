import argparse
import json
import logging
import os
import tarfile
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_model(model_path, test_data_path, output_path):
    if os.path.exists(os.path.join(model_path, "model.tar.gz")):
        with tarfile.open(os.path.join(model_path, "model.tar.gz"), "r:gz") as tar:
            tar.extractall(path=model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    df = pd.read_json(os.path.join(test_data_path, "test.jsonl"), lines=True)
    preds = sentiment_pipeline(df["text"].tolist(), batch_size=64)

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    predicted_labels = [label_map.get(p["label"].lower(), 1) for p in preds]
    acc = accuracy_score(df["label"].tolist(), predicted_labels)
    logging.info(f"✅ Accuracy: {acc:.4f}")

    os.makedirs(output_path, exist_ok=True)
    report = {"multiclass_classification_metrics": {"accuracy": {"value": acc}}}
    with open(os.path.join(output_path, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data_path, args.output_path)
