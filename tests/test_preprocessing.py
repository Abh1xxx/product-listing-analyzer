import unittest

from src.preprocessing import (
    clean_text,
    extract_ngrams,
    repeated_words,
    tokenize_sentences,
    tokenize_words,
)
from src.quality_score import score_listing


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_lowercases_removes_punctuation_and_normalizes_spaces(self):
        self.assertEqual(clean_text("  BLUE!! Cotton-shirt... "), "blue cotton shirt")

    def test_tokenization_and_sentence_count(self):
        self.assertEqual(tokenize_words("Soft, blue cotton!"), ["soft", "blue", "cotton"])
        self.assertEqual(len(tokenize_sentences("First sentence. Second sentence!")), 2)

    def test_repetition_and_ngrams(self):
        tokens = ["soft", "cotton", "soft", "shirt"]
        self.assertEqual(repeated_words(tokens), {"soft": 2})
        self.assertEqual(extract_ngrams(tokens, 2), ["soft cotton", "cotton soft", "soft shirt"])

    def test_invalid_ngram_size_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_ngrams(["word"], 0)


class QualityScoreTests(unittest.TestCase):
    def test_score_has_five_bounded_components_and_total(self):
        result = score_listing(
            "Blue cotton everyday shirt",
            "A comfortable cotton shirt for everyday use.",
            ["blue", "cotton", "everyday", "shirt"],
            ["comfortable", "cotton", "shirt", "for", "everyday", "use"],
            [("cotton", 0.5), ("shirt", 0.4)],
            {"color": ["blue"], "material": ["cotton"], "size": [], "features": []},
            ["color", "material", "size", "features"],
        )
        self.assertEqual(len(result["components"]), 5)
        self.assertEqual(result["total"], sum(result["components"].values()))
        self.assertTrue(all(0 <= points <= 20 for points in result["components"].values()))


if __name__ == "__main__":
    unittest.main()
