"""Shared text preprocessing and neural network architecture."""

import html
import re

from torch import nn


LABELS = ("negative", "positive")
MAX_FEATURES = 12_000
HIDDEN_DIM = 128
DROPOUT = 0.3
PREPROCESSING = "html-unescape-strip-tags-collapse-whitespace-v1"
HTML_TAGS = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    return " ".join(HTML_TAGS.sub(" ", html.unescape(text)).split())


class SentimentNetwork(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(LABELS)),
        )
