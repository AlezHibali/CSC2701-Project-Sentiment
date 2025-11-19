import argparse
import logging
import os
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_file_path, output_dir):
    logging.info(f"Reading data from {input_file_path}")
    df = pd.read_json(input_file_path, lines=True)
    logging.info(f"Initial data shape: {df.shape}")

    df.dropna(subset=["text", "label"], inplace=True)
    df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2])]
    logging.info(f"Data shape after cleaning: {df.shape}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train_clean.jsonl")
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    logging.info(f"✅ Saved cleaned data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--train-output", type=str, default="/opt/ml/processing/train")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.input_data_dir) if f.endswith(".jsonl")]
    if not files:
        raise FileNotFoundError("No .jsonl files found in input data directory.")
    preprocess_data(os.path.join(args.input_data_dir, files[0]), args.train_output)
