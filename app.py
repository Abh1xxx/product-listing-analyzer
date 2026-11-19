"""Streamlit interface for the Product Listing Analyzer."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import create_demo_dataset, load_dataset
from src.keyword_extraction import CorpusKeywordModel
from src.listing_analysis import analyze_listing, corpus_texts


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Product_Listing_Analyzer_500_Corpus_Detailed.xlsx"

st.set_page_config(page_title="Product Listing Analyzer", layout="wide")
st.title("Product Listing Analyzer")
st.write(
    "Explore product titles and descriptions with explainable NLP techniques. "
    "This academic demonstration runs locally and does not use an external AI service."
)

try:
    dataset, columns = load_dataset(DATA_PATH)
    using_demo_data = False
except FileNotFoundError as error:
    st.warning(f"{error} A small demonstration corpus is being used instead.")
    dataset = create_demo_dataset()
    columns = {
        "title": "Product_Title",
        "description": "Product_Description",
        "category": "Category",
        "color": "Color",
        "material": "Material",
        "size": "Size_or_Variant",
        "product_type": None,
        "features": ["Feature_1", "Feature_2"],
    }
    using_demo_data = True

keyword_model = CorpusKeywordModel.fit(corpus_texts(dataset, columns))

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Analyze listing", "Dataset overview", "Scoring guide"])
    st.caption("NLP fundamentals · local processing · rule-based score")

if page == "Dataset overview":
    st.header("Dataset overview")
    if using_demo_data:
        st.info("Showing the built-in demonstration corpus; the Excel dataset was not found.")
    st.metric("Product records", len(dataset))
    st.write("**Available columns**")
    st.code(", ".join(map(str, dataset.columns)))
    st.write("**Detected fields**")
    st.json(columns)
    st.write("**Column data types**")
    st.dataframe(
        pd.DataFrame(
            {"Column": dataset.columns, "Data type": dataset.dtypes.astype(str).values}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.write("**Preview**")
    st.dataframe(dataset.head(5), use_container_width=True, hide_index=True)
    st.write(
        f"Exact duplicate records removed when loading: "
        f"{int(dataset.attrs.get('duplicate_rows_removed', 0))}"
    )
elif page == "Scoring guide":
    st.header("How the educational score works")
    st.warning(
        "This project-defined metric is not an official marketplace ranking and has not "
        "been validated as a predictor of sales."
    )
    st.markdown(
        """
Each component contributes **up to 20 points**:

- **Title completeness:** 20 for at least 5 words; 12 for 3–4; 5 for 1–2; 0 if empty.
- **Keyword coverage:** 20 multiplied by the fraction of extracted corpus-TF-IDF keywords that also appear in the title.
- **Description quality:** up to 8 points for length (30+ words), 6 for sentence count (2+), and 6 for vocabulary diversity (10+ unique words). The lower thresholds earn half of each portion.
- **Attribute coverage:** 20 multiplied by detected dataset-backed attribute categories divided by categories assessable from available columns. An undetected value does **not** establish that the product lacks that attribute.
- **Repetition and readability:** start at 20; repeated-token ratios above 5%, 10%, and 20% subtract 2, 5, and 10 points. A title over 150 characters or a sentence over 35 words subtracts 5 points each.
"""
    )
else:
    st.header("Analyze a product listing")
    examples = {
        "Choose an example": ("", ""),
        "Cotton T-shirt": (
            "Northstar Blue Cotton T-Shirt for Everyday Wear",
            "This lightweight cotton T-shirt has a comfortable fit for everyday wear. "
            "The soft fabric and classic crew neck make it easy to pair with casual outfits.",
        ),
        "Wireless headphones": (
            "SoundPeak Black Wireless Headphones with Noise Isolation",
            "Enjoy clear audio with these lightweight wireless headphones. The padded ear "
            "cups provide a comfortable fit, while the rechargeable battery supports long listening sessions.",
        ),
    }
    example = st.selectbox("Example product input", list(examples))
    example_title, example_description = examples[example]
    title = st.text_input("Product title", value=example_title, max_chars=250)
    description = st.text_area(
        "Product description",
        value=example_description,
        height=160,
        max_chars=5000,
    )

    if st.button("Analyze listing", type="primary"):
        if not title.strip() or not description.strip():
            st.error("Enter both a product title and a product description.")
        elif not keyword_model.available:
            st.error("No usable title/description text is available in the corpus for TF-IDF.")
        else:
            try:
                result = analyze_listing(title, description, dataset, columns, keyword_model)
            except LookupError as error:
                st.error(str(error))
                st.info("NLTK needs its stopwords and WordNet data. Check your internet connection and retry.")
            else:
                score = result["quality_score"]
                st.subheader("Overall listing quality")
                st.metric("Project-defined score", f"{score['total']} / 100")
                st.caption(
                    "Educational rule-based score only; not an official marketplace ranking."
                )
                component_columns = st.columns(5)
                for column, (name, points) in zip(
                    component_columns, score["components"].items()
                ):
                    column.metric(name, f"{points} / 20")

                with st.expander("Original and cleaned text", expanded=True):
                    st.markdown(f"**Original title:** {result['original_title']}")
                    st.markdown(f"**Original description:** {result['original_description']}")
                    st.markdown(f"**Cleaned title:** `{result['cleaned_title']}`")
                    st.markdown(f"**Cleaned description:** `{result['cleaned_description']}`")

                left, right = st.columns(2)
                with left:
                    st.subheader("Tokenization and word forms")
                    st.write("**Tokens before stop-word removal:**", result["tokens_before_stopword_removal"])
                    st.write("**After stop-word removal:**", result["tokens_after_stopword_removal"])
                    st.write("**Porter stems:**", result["stemmed_tokens"])
                    st.write("**Lemmas:**", result["lemmatized_tokens"])
                    st.write(
                        f"Words: {result['word_count']} · Unique words: {result['unique_word_count']} · "
                        f"Characters: {result['character_count']} · Sentences: {result['sentence_count']}"
                    )
                with right:
                    st.subheader("Corpus TF-IDF keywords")
                    if result["keywords"]:
                        st.dataframe(
                            pd.DataFrame(result["keywords"], columns=["Keyword", "TF-IDF score"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No in-vocabulary terms were found.")
                    if result["cosine_similarity"] is not None:
                        st.write(f"Title–description cosine similarity: {result['cosine_similarity']:.3f}")

                lower_left, lower_right = st.columns(2)
                with lower_left:
                    st.subheader("Repetition and common phrases")
                    st.write("**Repeated words:**", result["repeated_words"] or "None")
                    st.write("**Most frequent content words:**", result["word_frequencies"])
                    st.write("**Common bigrams:**", result["bigrams"] or "None")
                    st.write("**Common trigrams:**", result["trigrams"] or "None")
                with lower_right:
                    st.subheader("Dataset-backed attributes")
                    for attribute, values in result["attributes"].items():
                        st.write(f"**{attribute.title()}:** {', '.join(values) if values else 'No match detected'}")
                    st.caption("No match means the rule did not find a known value; it does not prove an attribute is absent.")

                st.subheader("Problems and practical suggestions")
                if result["problems"]:
                    for problem in result["problems"]:
                        st.warning(problem)
                else:
                    st.success("No basic title-length, description-length, or repetition issue was detected.")
                for suggestion in result["suggestions"]:
                    st.write(f"- {suggestion}")

                with st.expander("Score calculation rules"):
                    for component, rule in score["rules"].items():
                        st.write(f"**{component}:** {rule}")
