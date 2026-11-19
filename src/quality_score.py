"""Transparent project-defined listing score (not a marketplace ranking)."""

from collections.abc import Sequence

from src.preprocessing import tokenize_sentences


def score_listing(
    title: str,
    description: str,
    title_tokens: Sequence[str],
    description_tokens: Sequence[str],
    keywords: Sequence[tuple[str, float]],
    detected_attributes: dict[str, list[str]],
    assessable_attributes: Sequence[str],
) -> dict[str, int | dict[str, str]]:
    """Score five components worth 20 points each, with explicit rule labels."""
    title_word_count = len(title_tokens)
    title_score = 20 if title_word_count >= 5 else 12 if title_word_count >= 3 else 5 if title_word_count else 0

    keyword_terms = [keyword for keyword, _ in keywords]
    covered_terms = sum(
        1 for keyword in keyword_terms if keyword.casefold() in title.casefold()
    )
    keyword_score = round(20 * covered_terms / len(keyword_terms)) if keyword_terms else 0

    description_word_count = len(description_tokens)
    description_sentences = len(tokenize_sentences(description))
    description_score = (
        (8 if description_word_count >= 30 else 4 if description_word_count >= 15 else 0)
        + (6 if description_sentences >= 2 else 3 if description_sentences == 1 else 0)
        + (6 if len(set(description_tokens)) >= 10 else 3 if len(set(description_tokens)) >= 5 else 0)
    )

    if assessable_attributes:
        found = sum(bool(detected_attributes.get(field)) for field in assessable_attributes)
        attribute_score = round(20 * found / len(assessable_attributes))
    else:
        attribute_score = 0

    all_tokens = list(title_tokens) + list(description_tokens)
    if all_tokens:
        from collections import Counter

        repeated_ratio = sum(count - 1 for count in Counter(all_tokens).values() if count > 1) / len(all_tokens)
    else:
        repeated_ratio = 0.0
    readability_score = 20
    if repeated_ratio > 0.20:
        readability_score -= 10
    elif repeated_ratio > 0.10:
        readability_score -= 5
    elif repeated_ratio > 0.05:
        readability_score -= 2
    if len(title) > 150:
        readability_score -= 5
    if max((len(sentence.split()) for sentence in description.split(".") if sentence.strip()), default=0) > 35:
        readability_score -= 5

    components = {
        "Title completeness": title_score,
        "Keyword coverage": keyword_score,
        "Description quality": description_score,
        "Attribute coverage": attribute_score,
        "Repetition and readability": readability_score,
    }
    return {
        "total": sum(components.values()),
        "components": components,
        "rules": {
            "Title completeness": "20 for 5+ words; 12 for 3–4; 5 for 1–2; otherwise 0.",
            "Keyword coverage": "20 × corpus-TF-IDF keywords also present in the title ÷ extracted keywords.",
            "Description quality": "8 for 30+ words (4 for 15–29); 6 for 2+ sentences (3 for 1); 6 for 10+ unique words (3 for 5–9).",
            "Attribute coverage": "20 × detected dataset-backed attribute categories ÷ categories available in the dataset; undetected does not mean the product lacks one.",
            "Repetition and readability": "Start at 20; subtract 2/5/10 for repeated-token ratios above 5%/10%/20%; subtract 5 for a title over 150 characters or a sentence over 35 words.",
        },
    }
