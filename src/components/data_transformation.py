import os
import sys
import re
import string
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "tfidf_vectorizer.pkl")


def clean_text(text: str) -> str:
    """Clean and normalize a single text string."""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{10,}\b", "", text)
    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = text.split()
        tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
        text = " ".join(tokens)

    return text


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logger.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Cleaning text data...")
            train_df["clean_text"] = train_df["text"].apply(clean_text)
            test_df["clean_text"] = test_df["text"].apply(clean_text)

            X_train = train_df["clean_text"].values
            y_train = train_df["label"].values
            X_test = test_df["clean_text"].values
            y_test = test_df["label"].values

            vectorizer = self.get_preprocessor()
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            save_object(self.config.preprocessor_path, vectorizer)
            logger.info(f"Vectorizer saved. Feature matrix: {X_train_tfidf.shape}")

            return X_train_tfidf, y_train, X_test_tfidf, y_test

        except Exception as e:
            raise CustomException(e, sys)
