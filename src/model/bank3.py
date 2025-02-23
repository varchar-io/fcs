#!/usr/bin/env python
import argparse
import logging
import os
import json

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
import joblib
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# --- Model and Data Collator Definitions ---


class TransactionClassifierFullFeatures(nn.Module):
    def __init__(
        self,
        transformer_model,
        hidden_size,
        num_labels,
        merchant_vocab_size,
        merchant_embed_dim,
    ):
        super(TransactionClassifierFullFeatures, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = hidden_size

        # Process merchant input: embed then project to hidden_size.
        self.merchant_embedding = nn.Embedding(merchant_vocab_size, merchant_embed_dim)
        self.merchant_fc = nn.Linear(merchant_embed_dim, hidden_size)

        # Process amount input: a small feed-forward network.
        self.amount_fc = nn.Sequential(
            nn.Linear(
                3, self.hidden_size // 2
            ),  # input now has 3 features: raw, sign, and log_abs
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )

        # Attention mechanism: compute a weight for each feature vector.
        # We expect three feature vectors (text, merchant, amount), each of dimension hidden_size.
        self.attention_vector = nn.Linear(hidden_size, 1)

        # Classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, merchant_id, amount, labels=None):
        device = input_ids.device
        if labels is not None:
            labels = labels.to(device)

        # Text representation from transformer (using CLS token)
        transformer_outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_repr = transformer_outputs.last_hidden_state[
            :, 0, :
        ]  # shape: (batch, hidden_size)
        text_repr = self.dropout(text_repr)

        # Merchant representation.
        merchant_emb = self.merchant_embedding(merchant_id)  # (batch, embed_dim)
        merchant_repr = self.merchant_fc(merchant_emb)  # (batch, hidden_size)

        # Amount representation. Use raw amount, the sign, and the logarithm of the absolute value; reshape to (batch, 3).
        # In forward(), process the amount input:
        amount = amount.view(-1, 1).float()
        sign = (amount >= 0).float()
        log_amount = torch.log(torch.abs(amount) + 1)
        amount_input = torch.cat([amount, sign, log_amount], dim=1)
        amount_repr = self.amount_fc(amount_input)

        # Stack features: shape (batch, 3, hidden_size)
        features = torch.stack([text_repr, merchant_repr, amount_repr], dim=1)

        # Compute attention scores for each feature.
        attn_scores = self.attention_vector(features)  # (batch, 3, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, 3, 1)
        combined = torch.sum(attn_weights * features, dim=1)  # (batch, hidden_size)

        logits = self.classifier(combined)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


def custom_data_collator(features, tokenizer):
    # Tokenize and pad text inputs.
    batch = tokenizer.pad(
        [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ],
        return_tensors="pt",
    )
    # Add labels if present.
    if "label" in features[0]:
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    # Add merchant_id if present.
    if "merchant_id" in features[0]:
        batch["merchant_id"] = torch.tensor(
            [f["merchant_id"] for f in features], dtype=torch.long
        )
    # Add amount if present.
    if "amount" in features[0]:
        batch["amount"] = torch.tensor(
            [f["amount"] for f in features], dtype=torch.float
        )
    return batch


# --- Pipeline Class Definition ---


class TransactionClassifierPipeline:
    def __init__(
        self,
        data_path="../data/cat_train.csv",
        output_dir="./final_model",
        model_checkpoint="distilbert-base-uncased",
        num_train_epochs=10,
        batch_size=16,
        max_length=128,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_checkpoint = model_checkpoint
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.config = AutoConfig.from_pretrained(self.model_checkpoint)
        self.hidden_size = self.config.hidden_size
        self.transformer_model = AutoModel.from_pretrained(
            self.model_checkpoint, config=self.config
        )

        # Placeholders for later
        self.model = None
        self.trainer = None
        self.dataset = None
        self.tokenized_dataset = None
        self.label2id = None
        self.id2label = None
        self.merchant2id = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(
            self.data_path,
            encoding="utf-8",
            names=["name", "merchant", "place", "flow", "amount", "category"],
        )
        # Filter: for example, keep only transactions with negative amount.
        df = df.head(150000)
        df = df.dropna(subset=["category"])
        # df["category"] = df["category"].str.split(".").str[0]

        # Label processing.
        labels = df["category"].unique()
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        df["label"] = df["category"].map(self.label2id)

        # Text for transformer input.
        df["text"] = df["name"].astype(str)

        # Process merchant as a categorical feature.
        merchants = df["merchant"].unique()
        self.merchant2id = {merchant: i for i, merchant in enumerate(merchants)}
        df["merchant_id"] = df["merchant"].map(self.merchant2id)

        # Create a dataset including text, merchant, amount and label.
        self.dataset = Dataset.from_pandas(
            df[["text", "merchant_id", "amount", "label"]]
        )

    def tokenize_dataset(self):
        # The tokenizer only handles text; pass through merchant_id and amount.
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            tokenized["merchant_id"] = examples["merchant_id"]
            tokenized["amount"] = examples["amount"]
            tokenized["label"] = examples["label"]
            return tokenized

        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        self.tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.3)

    def build_model(self):
        num_labels = len(self.label2id)
        merchant_vocab_size = len(self.merchant2id)
        # Set merchant embedding dimension (can be tuned)
        merchant_embed_dim = min(50, self.hidden_size // 2)
        self.model = TransactionClassifierFullFeatures(
            transformer_model=self.transformer_model,
            hidden_size=self.hidden_size,
            num_labels=num_labels,
            merchant_vocab_size=merchant_vocab_size,
            merchant_embed_dim=merchant_embed_dim,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_trainer(self):
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="weighted_f1",
            use_cpu=not torch.cuda.is_available(),
            max_grad_norm=1.0,
        )
        accuracy_metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            from sklearn.metrics import f1_score, precision_score, recall_score

            logits, labels = eval_pred
            predictions = logits.argmax(axis=1)
            accuracy = accuracy_metric.compute(
                predictions=predictions, references=labels
            )
            macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
            weighted_f1 = f1_score(
                labels, predictions, average="weighted", zero_division=0
            )
            precision = precision_score(
                labels, predictions, average="weighted", zero_division=0
            )
            recall = recall_score(
                labels, predictions, average="weighted", zero_division=0
            )
            return {
                "eval_accuracy": accuracy["accuracy"],
                "eval_macro_f1": macro_f1,
                "eval_weighted_f1": weighted_f1,
                "eval_precision": precision,
                "eval_recall": recall,
            }

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            data_collator=lambda features: custom_data_collator(
                features, self.tokenizer
            ),
            compute_metrics=compute_metrics,
        )

    def train(self):
        logging.info("Training with Text, Merchant, and Amount with attention:")
        self.trainer.train()
        train_results = self.trainer.evaluate(self.tokenized_dataset["train"])
        test_results = self.trainer.evaluate(self.tokenized_dataset["test"])
        logging.info("Test Accuracy: %.4f", test_results["eval_accuracy"])
        logging.info("Train Loss: %.4f", train_results["eval_loss"])

    def save(self):
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        joblib.dump(self.label2id, os.path.join(self.output_dir, "label2id.joblib"))
        joblib.dump(self.id2label, os.path.join(self.output_dir, "id2label.joblib"))
        joblib.dump(
            self.merchant2id, os.path.join(self.output_dir, "merchant2id.joblib")
        )
        logging.info("Model and artifacts saved successfully!")

    def load(self, model_dir):
        from safetensors.torch import load_file  # import safetensors

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.label2id = joblib.load(os.path.join(model_dir, "label2id.joblib"))
        self.id2label = joblib.load(os.path.join(model_dir, "id2label.joblib"))
        self.merchant2id = joblib.load(os.path.join(model_dir, "merchant2id.joblib"))

        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        safe_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=torch.device("cpu"))
        elif os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        else:
            raise FileNotFoundError("No model checkpoint found in the model directory.")

        self.build_model()
        self.model.load_state_dict(state_dict)
        self.model.to(torch.device("cpu"))
        logging.info("Model and artifacts loaded successfully!")

    def predict(self, texts, merchants, amounts):
        """
        Predict expects lists of texts, merchants, and amounts.
        """
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(merchants, list):
            merchants = [merchants]
        if not isinstance(amounts, list):
            amounts = [amounts]
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Convert merchant to id using the stored mapping.
        merchant_ids = [self.merchant2id.get(m, 0) for m in merchants]
        inputs["merchant_id"] = torch.tensor(merchant_ids, dtype=torch.long)
        inputs["amount"] = torch.tensor(amounts, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                merchant_id=inputs["merchant_id"],
                amount=inputs["amount"],
            )
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).tolist()
        return [self.id2label[p] for p in preds]


def run_training(model_dir="./final_model"):
    pipeline = TransactionClassifierPipeline(output_dir=model_dir)
    pipeline.load_and_preprocess_data()
    pipeline.tokenize_dataset()
    pipeline.build_model()
    pipeline.create_trainer()
    pipeline.train()
    pipeline.save()


def setup_pipeline_for_inference(model_dir="./final_model"):
    pipeline = TransactionClassifierPipeline(output_dir=model_dir)
    pipeline.load(model_dir)
    return pipeline


def run_interactive_inference(model_dir="./final_model"):
    logging.info("Inference mode: Running inference...")
    pipeline = setup_pipeline_for_inference(model_dir=model_dir)
    while True:
        text = input("Enter transaction description (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        merchant = input("Enter merchant: ")
        amount = float(input("Enter amount: "))
        prediction = pipeline.predict(texts=text, merchants=merchant, amounts=amount)
        logging.info(f"Category prediction: {prediction}")


def run_group_inference(model_dir="./final_model"):
    logging.info("Group Inference mode: Running inference...")
    pipeline = setup_pipeline_for_inference(model_dir=model_dir)
    df = pd.read_csv(
        "../data/cat_train.csv",
        encoding="utf-8",
        names=["name", "merchant", "place", "flow", "amount", "category"],
    )
    df = df.tail(10000)
    df = df.sample(n=10000)
    df["text"] = df["name"].astype(str)
    # Use the merchant and amount fields directly.
    df["merchant_id"] = df["merchant"]
    df["predicted_category"] = df.apply(
        lambda x: pipeline.predict(
            texts=x["text"], merchants=x["merchant"], amounts=x["amount"]
        )[0],
        axis=1,
    )
    df.to_csv("../data/cat_test_with_predictions.csv", index=False, encoding="utf-8")
    logging.info("Predictions saved to ../data/cat_test_with_predictions.csv")

    correct_predictions = sum(df["category"] == df["predicted_category"])
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    logging.info("Group Inference Accuracy: %.4f", accuracy)

    categories = sorted(df["category"].unique())
    precision, recall, _, _ = precision_recall_fscore_support(
        df["category"], df["predicted_category"], labels=categories
    )
    row_counts = (
        df["category"].value_counts().reindex(categories).fillna(0).astype(int).values
    )
    metrics_df = pd.DataFrame(
        {
            "Category": categories,
            "Precision": precision,
            "Recall": recall,
            "RowCount": row_counts,
        }
    )
    metrics_csv_path = "../data/cat_group_inference_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")
    print(f"Category metrics saved to {metrics_csv_path}")

    # New: Compute the confusion matrix and save to a different CSV file
    conf_mat = confusion_matrix(
        df["category"], df["predicted_category"], labels=categories
    )
    conf_mat_df = pd.DataFrame(conf_mat, index=categories, columns=categories)
    conf_mat_csv_path = "../data/cat_confusion_matrix.csv"
    conf_mat_df.to_csv(conf_mat_csv_path, encoding="utf-8")
    print(f"Confusion matrix saved to {conf_mat_csv_path}")

    # Find incorrect cases and group by (actual category, predicted category)
    incorrect_cases = df[df["category"] != df["predicted_category"]]
    grouped_incorrect = (
        incorrect_cases.groupby(["category", "predicted_category"])
        .size()
        .reset_index(name="count")
    )
    grouped_incorrect = grouped_incorrect.sort_values(by="count", ascending=False)
    incorrect_cases_csv_path = "../data/cat_incorrect_cases.csv"
    grouped_incorrect.to_csv(incorrect_cases_csv_path, index=False, encoding="utf-8")
    print(
        f"Incorrect cases grouped by (actual category, predicted category) saved to {incorrect_cases_csv_path}"
    )

    # New: Compute accuracy by merchant for those with 10+ rows
    merchant_group = df.groupby("merchant")
    merchant_counts = merchant_group.size()
    merchant_accuracy = merchant_group.apply(
        lambda x: (x["category"] == x["predicted_category"]).mean(),
        include_groups=False,
    )
    merchant_df = pd.DataFrame(
        {
            "Merchant": merchant_accuracy.index,
            "RowCount": merchant_counts.values,
            "Accuracy": merchant_accuracy.values,
        }
    )
    merchant_df = merchant_df[merchant_df["RowCount"] >= 10]
    merchant_csv_path = "../data/cat_accuracy_by_merchant.csv"
    merchant_df.to_csv(merchant_csv_path, index=False, encoding="utf-8")
    print(f"Accuracy by merchant saved to {merchant_csv_path}")


# flag to load the model once at service start
loaded = False
model_dir = "./src/data/fcs_v3"


# model will be placed in `../data/fcs-v3` folder
def classify(input):
    global loaded, pipeline
    if not loaded:
        # seems there is a segment fault in torch
        torch.set_num_threads(1)

        # load the model
        loaded = True
        pipeline = setup_pipeline_for_inference(model_dir=model_dir)

    # local manual test: name, merchant and amount are required
    # input = [
    #     {
    #         "name": "REVENUE CONTROL 9800 Airport Blvd",
    #         "merchant": "REVENUE CONTROL",
    #         "amount": -1,
    #     },
    # ]

    # convert input to dataframe
    df = pd.DataFrame(input)

    # add merchant column if it doesn't exist
    if "name" not in df.columns:
        df["name"] = ""

    if "merchant" not in df.columns:
        df["merchant"] = ""

    # add amount column (default to expense) if it doesn't exist
    if "amount" not in df.columns:
        df["amount"] = 0.0

    # Use the pipeline to predict categories.
    categories = pipeline.predict(
        texts=df["name"].tolist(),
        merchants=df["merchant"].tolist(),
        amounts=df["amount"].tolist(),
    )

    # print(categories)
    result = json.dumps(categories)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dryrun", action="store_true", help="Dry run the inference function"
    )
    parser.add_argument(
        "-t", "--training", action="store_true", help="Run training mode"
    )
    parser.add_argument(
        "-i", "--inference", action="store_true", help="Run inference mode"
    )
    parser.add_argument(
        "-g", "--group", action="store_true", help="Run group inference mode"
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        default="./final_model",
        help="Output directory for the model",
    )
    args = parser.parse_args()

    if args.training:
        run_training(model_dir=args.model_dir)
    elif args.inference:
        run_interactive_inference(model_dir=args.model_dir)
    elif args.group:
        run_group_inference(model_dir=args.model_dir)
    elif args.dryrun:
        classify(None)
    else:
        parser.print_help()
