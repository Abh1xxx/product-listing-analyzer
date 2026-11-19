"""Coordinate preprocessing, corpus keywords, attributes, and listing advice."""

from collections import Counter

import pandas as pd

from src.attribute_detection import available_attribute_fields, detect_attributes
from src.keyword_extraction import CorpusKeywordModel
from src.preprocessing import (
    clean_text,
    extract_ngrams,
    lemmatize_tokens,
    remove_stop_words,
    repeated_words,
    stem_tokens,
    tokenize_sentences,
    tokenize_words,
)
from src.quality_score import score_listing


def corpus_texts(frame: pd.DataFrame, columns: dict) -> list[str]:
    """Combine available title and description columns for corpus modeling."""
    text_columns = [columns.get("title"), columns.get("description")]
    valid_columns = [column for column in text_columns if column and column in frame]
    if not valid_columns:
        return []
    return (
        frame[valid_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )


def analyze_listing(
    title: str,
    description: str,
    frame: pd.DataFrame,
    columns: dict,
    keyword_model: CorpusKeywordModel,
) -> dict:
    """Analyze a manually entered title and description."""
    combined_text = f"{title.strip()} {description.strip()}".strip()
    title_tokens = tokenize_words(title)
    description_tokens = tokenize_words(description)
    tokens = tokenize_words(combined_text)
    content_tokens = remove_stop_words(tokens)
    keywords = keyword_model.keywords(combined_text)
    attributes = detect_attributes(combined_text, frame, columns)

    repeated = repeated_words(tokens)
    suggestions = []
    problems = []
    if len(title_tokens) < 5:
        problems.append("The title is short (fewer than five words).")
        suggestions.append("Add useful details such as product type, material, color, or size when applicable.")
    if len(description_tokens) < 20:
        problems.append("The description is very short (fewer than 20 words).")
        suggestions.append("Expand the description with clear, factual details about use, construction, and features.")
    if any(count > 2 for count in repeated.values()):
        problems.append("Some words are repeated more than twice.")
        suggestions.append("Remove repeated words and keep each product benefit concise.")
    if keywords:
        title_keyword_count = sum(keyword.casefold() in title.casefold() for keyword, _ in keywords)
        if title_keyword_count < min(3, len(keywords)):
            suggestions.append("Consider including the most relevant description terms in the title without keyword stuffing.")
    if not any(attributes.values()):
        suggestions.append("If relevant, state product attributes clearly; this rule-based detector only recognizes values found in the dataset.")
    if not suggestions:
        suggestions.append("The listing has useful detail. Keep all claims accurate and specific to the product.")

    score = score_listing(
        title,
        description,
        title_tokens,
        description_tokens,
        keywords,
        attributes,
        available_attribute_fields(columns),
    )
    top_bigrams = Counter(extract_ngrams(content_tokens, 2)).most_common(5)
    top_trigrams = Counter(extract_ngrams(content_tokens, 3)).most_common(5)

    return {
        "original_title": title,
        "original_description": description,
        "cleaned_title": clean_text(title),
        "cleaned_description": clean_text(description),
        "tokens_before_stopword_removal": tokens,
        "tokens_after_stopword_removal": content_tokens,
        "stemmed_tokens": stem_tokens(content_tokens),
        "lemmatized_tokens": lemmatize_tokens(content_tokens),
        "word_count": len(tokens),
        "character_count": len(combined_text),
        "sentence_count": len(tokenize_sentences(description)),
        "unique_word_count": len(set(tokens)),
        "repeated_words": repeated,
        "word_frequencies": Counter(content_tokens).most_common(15),
        "keywords": keywords,
        "bigrams": top_bigrams,
        "trigrams": top_trigrams,
        "attributes": attributes,
        "problems": problems,
        "suggestions": suggestions,
        "quality_score": score,
        "cosine_similarity": keyword_model.title_description_similarity(title, description),
    }
