# IMDb Sentiment Neural Network

Binary sentiment classification for English IMDb movie reviews. The pipeline learns a
TF-IDF representation and a small PyTorch neural network, evaluates on held-out reviews,
and publishes the model and metrics to Hugging Face from GitHub Actions.

[Published Hugging Face model](https://huggingface.co/Fetty0707/imdb-sentiment-nn-course-10k) · [GitHub Actions runs](https://github.com/YumemoeNova/imdb-sentiment-nn/actions/workflows/train-and-upload.yml)

## Data and model

The input is the [course-provided `imdb_balanced_10k.csv`](https://gitee.com/lundechen/data_engineering_master_2026_fall/blob/cf10d7b5c0ba417c76723e34de49d3512766748e/session-400/datasets/imdb_balanced_10k.csv)
from `session-400/datasets/`. It contains 10,000 labeled reviews, with 5,000 reviews
per class (`0` negative and `1` positive). `data/prepare_data.py` fetches the file
at the pinned course Git revision and verifies its SHA-256. Raw data and trained
files are excluded from this project's Git repository.

Text is HTML-decoded and whitespace is normalized. After 17 duplicate reviews are
removed, a stratified 80/20 split with seed 42 yields 7,986 training reviews and
1,997 held-out test reviews. TF-IDF learns up to 12,000
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

Install Git before running the preparation script. If you already have the course
repository checked out, you can also try the trainer against its local CSV with
`python train.py --data PATH/TO/session-400/datasets/imdb_balanced_10k.csv`.
The trainer also accepts the optional `imdb_top_500.csv` through `--data`. The
workflow always fetches and verifies the pinned 10K course file.

## Artifacts

| File | Contents |
| --- | --- |
| `model.pt` | PyTorch state dictionary |
| `vectorizer.pkl` | Fitted scikit-learn TF-IDF vectorizer |
| `config.json` | Network settings, labels, dependencies, course source and data hash, and project Git commit |
| `metrics.json` | Held-out metrics and sample counts |

To use this project's published model, run the following from the repository root.
Add `--revision COMMIT` to select a particular published version.

```sh
python predict.py --hf-repo Fetty0707/imdb-sentiment-nn-course-10k "A surprisingly moving story."
```

Only load `vectorizer.pkl` from projects you trust. Predictions are intended for
English movie reviews, and the softmax probabilities are not calibrated.

## GitHub Actions and Hugging Face

1. Create a [Hugging Face access token](https://huggingface.co/settings/tokens) with write permission.
2. Open the project's **Settings → Secrets and variables → Actions** on GitHub and add a
   repository secret named `HF_TOKEN`. The workflow defaults to the token owner's
   `imdb-sentiment-nn-course-10k` model repository and creates it as public on the first run.
   To point at a different model repository, set an Actions repository variable named `HF_REPO_ID`
   to `owner/name`. If using a token scoped to a single existing repository, create an empty
   public model repository first and grant the token write access to it.
3. Push to `main` or `master`, or select **Run workflow** from the Actions tab after the
   initial setup. The runner installs dependencies, fetches and verifies the course CSV,
   trains, checks inference, archives the four files in GitHub Actions, and publishes
   them with a generated model card.
   A successful run includes the Hugging Face URL and commit in its summary.

Subsequent changes are pushed to GitHub and produce a fresh Hugging Face model revision.
A first run made before `HF_TOKEN` is configured can still produce downloadable Actions
artifacts. Start a new run after adding the secret to publish the model.
