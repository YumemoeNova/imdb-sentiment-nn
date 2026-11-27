"""Classify a single English IMDb-style review with saved artifacts."""

import argparse
import json
from pathlib import Path

import joblib
import torch
from huggingface_hub import snapshot_download

from model import LABELS, PREPROCESSING, SentimentNetwork, clean_text


INFERENCE_FILES = ("model.pt", "vectorizer.pkl", "config.json")


def predict(text: str, model_dir: Path) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Provide a non-empty review.")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config["labels"] != list(LABELS) or config["preprocessing"] != PREPROCESSING:
        raise ValueError("The artifacts are incompatible with this version of the prediction code.")
    network = SentimentNetwork(config["input_dim"], config["hidden_dim"], config["dropout"])
    network.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    network.eval()

    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    features = torch.from_numpy(vectorizer.transform([cleaned]).toarray())
    with torch.inference_mode():
        scores = network(features).softmax(dim=1)[0]
    return {
        "label": LABELS[scores.argmax().item()],
        "probabilities": dict(zip(LABELS, scores.tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="English review to classify.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    source.add_argument("--hf-repo", help="Trusted Hugging Face model repository as owner/name.")
    parser.add_argument("--revision", help="Hugging Face Git branch, tag or commit to load.")
    args = parser.parse_args()
    if args.revision and not args.hf_repo:
        parser.error("--revision requires --hf-repo")
    if not clean_text(args.text):
        parser.error("Provide a non-empty review.")

    model_dir = args.model_dir
    if args.hf_repo:
        model_dir = Path(snapshot_download(
            repo_id=args.hf_repo, revision=args.revision, allow_patterns=list(INFERENCE_FILES),
        ))
    missing = [name for name in INFERENCE_FILES if not (model_dir / name).is_file()]
    if missing:
        parser.error(f"Missing {', '.join(missing)} in {model_dir}. Train or supply a model first.")
    print(json.dumps(predict(args.text, model_dir), indent=2))


if __name__ == "__main__":
    main()
