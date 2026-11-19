"""Corpus-fitted TF-IDF keyword extraction and optional similarity."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CorpusKeywordModel:
    vectorizer: TfidfVectorizer | None
    available: bool

    @classmethod
    def fit(cls, corpus: list[str]) -> "CorpusKeywordModel":
        usable_texts = [text for text in corpus if text.strip()]
        if not usable_texts:
            return cls(vectorizer=None, available=False)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            vectorizer.fit(usable_texts)
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise
            return cls(vectorizer=None, available=False)
        return cls(vectorizer=vectorizer, available=True)

    def keywords(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Rank terms in this listing using IDF learned from the full corpus."""
        if not self.available or self.vectorizer is None or not text.strip():
            return []
        row = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = row.toarray()[0]
        ranked_indices = scores.argsort()[::-1]
        return [
            (feature_names[index], float(scores[index]))
            for index in ranked_indices
            if scores[index] > 0
        ][:limit]

    def title_description_similarity(self, title: str, description: str) -> float | None:
        """Return cosine similarity for the two texts using the fitted corpus space."""
        if (
            not self.available
            or self.vectorizer is None
            or not title.strip()
            or not description.strip()
        ):
            return None
        vectors = self.vectorizer.transform([title, description])
        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])
