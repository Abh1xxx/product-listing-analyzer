"""Bundle the required NLTK corpora during the Vercel build."""

from pathlib import Path

import nltk

destination = Path(__file__).resolve().parents[1] / "nltk_data"
destination.mkdir(exist_ok=True)
for resource in ("stopwords", "wordnet"):
    if not nltk.download(resource, download_dir=str(destination), quiet=True):
        raise RuntimeError(f"Could not download required NLTK resource: {resource}")
