import argparse
import logging
import os
import tarfile
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main(args):
    # Resolve pretrained model (from Registry or fallback)
    model_source = args.model_name
    if args.pretrained_dir:
        tar_path = os.path.join(args.pretrained_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(args.pretrained_dir)
            for root, _, files in os.walk(args.pretrained_dir):
                if "config.json" in files:
                    model_source = root
                    break
            logging.info(f"Using Registry pretrained weights from {model_source}")
        else:
            logging.warning("No model.tar.gz found, using default model_name.")

    # Load data directly from JSONL
    train_path = os.path.join(args.train_dir, "train_clean.jsonl")
    val_path = os.path.join(args.validation_dir, "validation.jsonl")
    train_dataset = load_dataset("json", data_files=train_path)["train"]
    eval_dataset = load_dataset("json", data_files=val_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=3)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    logging.info(f"✅ Model saved to {args.model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--pretrained_dir", type=str, default=os.environ.get("SM_CHANNEL_PRETRAINED"))
    args = parser.parse_args()
    main(args)