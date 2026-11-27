"""Train and evaluate a TF-IDF feedforward network for IMDb reviews."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn

from model import DROPOUT, HIDDEN_DIM, LABELS, MAX_FEATURES, PREPROCESSING
from model import SentimentNetwork, clean_text


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "imdb_balanced_10k.csv"
ARTIFACTS = ("model.pt", "vectorizer.pkl", "config.json", "metrics.json")
LABEL_IDS = {"0": 0, "1": 1, "neg": 0, "pos": 1, "negative": 0, "positive": 1}
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def load_reviews(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = {name.strip().lower(): name for name in reader.fieldnames or []}
        text_column = columns.get("text") or columns.get("review")
        label_column = columns.get("label") or columns.get("sentiment")
        split_column = columns.get("split")
        if not text_column or not label_column:
            raise ValueError("Expected text/review and label/sentiment columns.")

        groups = {"train": [], "test": []} if split_column else {"all": []}
        for row in reader:
            text = clean_text(row[text_column] or "")
            label = LABEL_IDS.get((row[label_column] or "").strip().lower())
            split = (row[split_column] or "").strip().lower() if split_column else "all"
            if not text or label is None or split not in groups:
                raise ValueError(f"Invalid text, label or split on CSV line {reader.line_num}.")
            groups[split].append((text, label))

    # Process train first so a duplicated test review is excluded from evaluation.
    seen = {}
    unique = {split: [] for split in groups}
    for split, reviews in groups.items():
        for text, label in reviews:
            key = text.casefold()
            if key in seen:
                if seen[key] != label:
                    raise ValueError("Duplicate reviews have conflicting labels.")
                continue
            seen[key] = label
            unique[split].append((text, label))

    if split_column:
        train, test = unique["train"], unique["test"]
    else:
        train, test = train_test_split(
            unique["all"], test_size=0.2, random_state=SEED,
            stratify=[label for _, label in unique["all"]],
        )
    for split, rows in (("train", train), ("test", test)):
        if {label for _, label in rows} != {0, 1}:
            raise ValueError(f"The {split} split must contain both sentiment classes.")
    return train, test


def batch_logits(network, features, indices):
    return network(torch.from_numpy(features[indices].toarray()))


def train(data: Path, output_dir: Path, epochs: int) -> dict:
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rng = np.random.default_rng(SEED)
    train_rows, test_rows = load_reviews(data)
    train_texts, train_labels = zip(*train_rows)
    test_texts, test_labels = zip(*test_rows)

    # Fit only on the train split and keep TF-IDF sparse until each PyTorch batch.
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2,
        strip_accents="unicode", dtype=np.float32, sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    targets = torch.tensor(train_labels, dtype=torch.long)

    config = {
        "input_dim": train_features.shape[1],
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "labels": list(LABELS),
        "preprocessing": PREPROCESSING,
        "training": {
            "seed": SEED, "epochs": epochs, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "versions": {"torch": torch.__version__, "scikit_learn": sklearn.__version__},
        "dataset": {"filename": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        "git_commit": os.environ.get("GITHUB_SHA"),
    }
    network = SentimentNetwork(config["input_dim"], HIDDEN_DIM, DROPOUT)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    network.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(batch_logits(network, train_features, indices), targets[indices])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        print(f"Epoch {epoch:2}/{epochs} | train loss {loss_sum / len(order):.4f}")

    predictions = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(test_rows), BATCH_SIZE):
            indices = slice(start, start + BATCH_SIZE)
            predictions.extend(batch_logits(network, test_features, indices).argmax(dim=1).tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, labels=[0, 1], average="macro", zero_division=0,
    )
    metrics = {
        "test_accuracy": float(accuracy_score(test_labels, predictions)),
        "test_macro_f1": float(f1),
        "test_macro_precision": float(precision),
        "test_macro_recall": float(recall),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output_dir / "model.pt")
    joblib.dump(vectorizer, output_dir / "vectorizer.pkl")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. Run python data/prepare_data.py first.")
    print(json.dumps(train(args.data, args.output_dir, args.epochs), indent=2))


if __name__ == "__main__":
    main()
