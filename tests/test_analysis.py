from pathlib import Path
import unittest

from src.data_loader import identify_columns, load_dataset
from src.keyword_extraction import CorpusKeywordModel
from src.listing_analysis import analyze_listing, corpus_texts


class DatasetAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "Product_Listing_Analyzer_500_Corpus_Detailed.xlsx"
        )
        cls.frame, cls.columns = load_dataset(dataset_path)
        cls.model = CorpusKeywordModel.fit(corpus_texts(cls.frame, cls.columns))

    def test_inspected_workbook_schema_is_used(self):
        self.assertEqual(len(self.frame), 500)
        self.assertEqual(self.columns["title"], "Product_Title")
        self.assertEqual(self.columns["description"], "Product_Description")
        self.assertEqual(self.columns["features"], ["Feature_1", "Feature_2"])

    def test_partial_column_schemas_are_supported(self):
        columns = identify_columns(["Title", "Description", "Color"])
        self.assertEqual(columns["title"], "Title")
        self.assertEqual(columns["description"], "Description")
        self.assertEqual(columns["color"], "Color")
        self.assertIsNone(columns["material"])
        self.assertEqual(columns["features"], [])

    def test_corpus_with_no_tfidf_vocabulary_is_reported_unavailable(self):
        model = CorpusKeywordModel.fit(["and the it", "or the a"])
        self.assertFalse(model.available)
        self.assertEqual(model.keywords("and the"), [])

    def test_listing_analysis_uses_corpus_tfidf_and_known_attributes(self):
        result = analyze_listing(
            "TrailCore Beige Sandals for Unisex Size 8 US",
            "Comfortable beige leather sandals with a practical slip-on design for everyday use.",
            self.frame,
            self.columns,
            self.model,
        )
        self.assertTrue(result["keywords"])
        self.assertIn("beige", [keyword for keyword, _ in result["keywords"]])
        self.assertEqual(result["attributes"]["color"], ["beige"])
        self.assertEqual(result["attributes"]["material"], ["leather"])
        self.assertEqual(result["attributes"]["size"], ["8 US"])
        self.assertEqual(sum(result["quality_score"]["components"].values()), result["quality_score"]["total"])


if __name__ == "__main__":
    unittest.main()
