"""Build a balanced CSV from the pinned Stanford IMDb dataset."""

import argparse
import csv
import io
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import pyarrow.parquet as parquet


DATASET = "stanfordnlp/imdb"
REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
SEED = 42
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("imdb_balanced_10k.csv")
SPLIT_COUNTS = {"train": 4_000, "test": 1_000}


def prepare(output: Path) -> None:
    if output.exists():
        print(f"Using existing dataset at {output}")
        return

    rng = random.Random(SEED)
    samples = []
    for split, count in SPLIT_COUNTS.items():
        filename = f"plain_text/{split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}"
        with urlopen(url, timeout=120) as response:
            table = parquet.read_table(io.BytesIO(response.read()), columns=["text", "label"])
        reviews = {0: [], 1: []}
        for row in table.to_pylist():
            reviews[row["label"]].append(row["text"])
        for label, texts in reviews.items():
            samples.extend((text, label, split) for text in rng.sample(texts, count))

    rng.shuffle(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with staged.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\r\n")
            writer.writerow(("text", "label", "split"))
            writer.writerows(samples)
        staged.replace(output)
    print(f"Prepared {len(samples):,} balanced reviews at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
