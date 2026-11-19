"""Beginner-friendly text preprocessing using NLTK."""

from collections import Counter
import re
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer


_word_tokenizer = TreebankWordTokenizer()
_sentence_tokenizer = PunktSentenceTokenizer()
_stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase text, replace punctuation with spaces, and normalize whitespace."""
    lowercase_text = str(text or "").lower()
    without_punctuation = re.sub(r"[^\w\s]", " ", lowercase_text)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def tokenize_words(text: str) -> list[str]:
    """Tokenize words with NLTK after removing punctuation."""
    return [
        token.lower()
        for token in _word_tokenizer.tokenize(clean_text(text))
        if re.search(r"\w", token)
    ]


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt tokenizer."""
    return [sentence.strip() for sentence in _sentence_tokenizer.tokenize(str(text or "")) if sentence.strip()]


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    """Load NLTK English stop words, downloading the corpus if it is missing."""
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise LookupError(
                "NLTK stop words are unavailable. Run "
                "`python -m nltk.downloader stopwords` and try again."
            )
        return set(nltk.corpus.stopwords.words("english"))


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove common English words that usually carry little topic information."""
    stop_words = get_stop_words()
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Reduce words to approximate roots with the Porter stemming algorithm."""
    return [_stemmer.stem(token) for token in tokens]


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise LookupError(
                "NLTK WordNet data is unavailable. Run "
                "`python -m nltk.downloader wordnet` and try again."
            )
    return WordNetLemmatizer()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Convert words to dictionary forms (noun form unless otherwise specified)."""
    lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def word_frequencies(tokens: list[str]) -> Counter[str]:
    """Count how often each token occurs."""
    return Counter(tokens)


def repeated_words(tokens: list[str]) -> dict[str, int]:
    """Return words appearing at least twice, sorted by frequency."""
    return dict(
        sorted(
            ((word, count) for word, count in word_frequencies(tokens).items() if count > 1),
            key=lambda item: (-item[1], item[0]),
        )
    )


def extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Create adjacent word sequences of length n."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
