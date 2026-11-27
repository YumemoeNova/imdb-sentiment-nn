"""Publish the GitHub Actions training artifacts as one Hugging Face commit."""

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi

from train import ARTIFACTS, ROOT


MODEL_NAME = "imdb-sentiment-nn-course-10k"


def model_card(config: dict, metrics: dict, github_repo: str, run_id: str) -> str:
    source = config["git_commit"]
    dataset = config["dataset"]
    dataset_description = f"The training CSV was `{dataset['filename']}`."
    if "source_url" in dataset:
        dataset_description = (
            f"Training used the [course CSV `{dataset['filename']}`]({dataset['source_url']}). "
            "After deduplication, it was stratified into an 80% training and 20% test split with seed 42."
        )
    return f"""---
language:
- en
pipeline_tag: text-classification
tags:
- sentiment-analysis
- pytorch
- tf-idf
- imdb
---

# IMDb Sentiment Neural Network

This CPU model classifies English IMDb-style movie reviews as positive or negative.
It combines word unigram and bigram TF-IDF with a one-hidden-layer PyTorch network.

- [Source code and inference instructions](https://github.com/{github_repo}/tree/{source})
- [GitHub Actions training run](https://github.com/{github_repo}/actions/runs/{run_id})

## Evaluation

{dataset_description}

The held-out test set contains {metrics['test_samples']:,} reviews. Training used
{metrics['train_samples']:,} reviews. The TF-IDF vocabulary was fitted on the training split.

| Metric | Value |
| --- | ---: |
| Accuracy | {metrics['test_accuracy']:.4f} |
| Macro F1 | {metrics['test_macro_f1']:.4f} |
| Macro precision | {metrics['test_macro_precision']:.4f} |
| Macro recall | {metrics['test_macro_recall']:.4f} |

The input SHA-256 and any pinned source revision are recorded in `config.json`.
Use the linked repository's `predict.py` for inference. The softmax scores are not
calibrated. Accuracy may be lower for sarcasm, mixed opinions, other languages,
and text outside the movie-review domain.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Missing HF_TOKEN. Add it under the GitHub repository's Actions secrets.")

    directory = ROOT / "model"
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if config["git_commit"] != os.environ["GITHUB_SHA"]:
        raise SystemExit("The model artifacts do not match the source commit for this run.")

    api = HfApi(token=token)
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        repo_id = f"{api.whoami()['name']}/{MODEL_NAME}"

    with TemporaryDirectory() as temp_dir:
        folder = Path(temp_dir)
        for filename in ARTIFACTS:
            shutil.copyfile(directory / filename, folder / filename)
        card = model_card(config, metrics, os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"])
        (folder / "README.md").write_text(card, encoding="utf-8")
        if not api.repo_exists(repo_id, repo_type="model"):
            api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=folder,
            commit_message=f"Publish model from GitHub {config['git_commit'][:12]}",
        )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"Published {repo_url} at {commit.oid}")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write(f"### Hugging Face model\n\n[{repo_id}]({repo_url}) · [{commit.oid[:12]}]({commit.commit_url})\n")


if __name__ == "__main__":
    main()
