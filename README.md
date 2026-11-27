# IMDb Sentiment Neural Network

Binary sentiment classification for English IMDb movie reviews. The pipeline learns a
TF-IDF representation and a small PyTorch neural network, evaluates on held-out reviews,
and publishes the model and metrics to Hugging Face from GitHub Actions.

[GitHub Actions runs](https://github.com/YumemoeNova/imdb-sentiment-nn/actions/workflows/train-and-upload.yml)

## Data and model

The project defaults to a reproducible `data/imdb_balanced_10k.csv` generated from
[Stanford's Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
([Maas et al., 2011](https://aclanthology.org/P11-1015/)). The course CSVs were absent
from the provided materials, so this is a separately sampled dataset with the same
size and balance as the named 10K option. `data/prepare_data.py` uses a pinned revision
of [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) and seed 42 to
sample 4,000 reviews per label from the official training split and 1,000 per label
from the official test split. Raw data and trained files are excluded from Git.

Text is HTML-decoded and whitespace is normalized. Duplicate reviews are removed
before training, keeping each review in a single split. TF-IDF learns up to 12,000
one- and two-word features from the training set. The network uses a 128-unit dense
layer, ReLU, dropout, and two output logits. Training uses cross-entropy and Adam.
Evaluation reports accuracy and macro-averaged F1, precision, and recall.

## Run locally

Python 3.11 is used in CI. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe data/prepare_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py "I loved the movie and the acting was fantastic."
```

On Linux or macOS with Python 3.11, use `python3.11 -m venv .venv`, then replace
`.\.venv\Scripts\python.exe` with `.venv/bin/python` in subsequent commands.

To use an original course CSV, place it in `data/` and run
`python train.py --data data/imdb_balanced_10k.csv` or
`python train.py --data data/imdb_top_500.csv` with the environment activated.
The trainer accepts `text` or `review` and `label` or `sentiment` columns, with
0/1 or negative/positive labels. A `split` column with train/test values is optional.
Without one, training uses a deterministic stratified 80/20 split. For CI, the input
CSV must be available to the runner. If the course allows redistribution, commit the
provided 10K CSV at the default path to have the preparation step use it as-is.
An authorized external dataset URL can also be integrated into the preparation step.

## Artifacts

| File | Contents |
| --- | --- |
| `model.pt` | PyTorch state dictionary |
| `vectorizer.pkl` | Fitted scikit-learn TF-IDF vectorizer |
| `config.json` | Network settings, labels, dependencies, data hash and Git commit |
| `metrics.json` | Held-out metrics and sample counts |

For a published model, use the same project code and replace `OWNER` with the Hugging Face username.
Add `--revision COMMIT` to select a particular published version.

```sh
python predict.py --hf-repo OWNER/imdb-sentiment-nn "A surprisingly moving story."
```

Only load `vectorizer.pkl` from projects you trust. Predictions are intended for
English movie reviews, and the softmax probabilities are not calibrated.

## GitHub Actions and Hugging Face

1. Create a [Hugging Face access token](https://huggingface.co/settings/tokens) with write permission.
2. Open the project's **Settings → Secrets and variables → Actions** on GitHub and add a
   repository secret named `HF_TOKEN`. The workflow defaults to the token owner's
   `imdb-sentiment-nn` model repository and creates it as public on the first run.
   To point at a different model repository, set an Actions repository variable named `HF_REPO_ID`
   to `owner/name`. If using a token scoped to a single existing repository, create an empty
   public model repository first and grant the token write access to it.
3. Push to `main` or `master`, or select **Run workflow** from the Actions tab after the
   initial setup. The runner installs dependencies, prepares data, trains, verifies inference,
   archives the four files in GitHub Actions, and publishes them with a generated model card.
   A successful run includes the Hugging Face URL and commit in its summary.

Subsequent changes are pushed to GitHub and produce a fresh Hugging Face model revision.
A first run made before `HF_TOKEN` is configured can still produce downloadable Actions
artifacts. Start a new run after adding the secret to publish the model.
