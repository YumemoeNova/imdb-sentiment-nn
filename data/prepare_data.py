"""Retrieve and verify the exact IMDb CSV distributed with the course."""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


COURSE_REPO = "https://gitee.com/lundechen/data_engineering_master_2026_fall"
COURSE_REVISION = "cf10d7b5c0ba417c76723e34de49d3512766748e"
COURSE_PATH = "session-400/datasets/imdb_balanced_10k.csv"
DATASET_URL = f"{COURSE_REPO}/blob/{COURSE_REVISION}/{COURSE_PATH}"
DATASET_SHA256 = "02fe0c2bbd68cdb6998d45e7a64f216e455b0927db9241dd784dfcdb40341572"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name(Path(COURSE_PATH).name)


def prepare(output: Path) -> None:
    if output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() == DATASET_SHA256:
        print(f"Using verified course dataset at {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        repo = work / "course"
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True, env=env)
        # Gitee requires a login for large raw HTTP downloads. Fetch through public Git.
        subprocess.run(
            [
                "git", "-c", "credential.helper=", "-C", str(repo),
                "fetch", "--quiet", "--no-tags", "--depth=1", f"{COURSE_REPO}.git", COURSE_REVISION,
            ],
            check=True, env=env, timeout=180,
        )
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"FETCH_HEAD:{COURSE_PATH}"],
            env=env, timeout=30,
        )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != DATASET_SHA256:
            raise ValueError(f"Course CSV SHA-256 mismatch: expected {DATASET_SHA256}, got {digest}")

        staged = work / output.name
        staged.write_bytes(raw)
        staged.replace(output)
    print(f"Prepared verified course dataset at {output} (SHA-256 {DATASET_SHA256})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
