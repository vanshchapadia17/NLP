import os
import sys
import re
import string
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

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


@dataclass
class Word2VecTransformationConfig:
    model_path: str = os.path.join("artifacts", "word2vec_model.pkl")


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


class Word2VecTransformation:
    def __init__(self):
        self.config = Word2VecTransformationConfig()
        self.vector_size = 100

    def _text_to_vector(self, text: str, wv) -> np.ndarray:
        words = text.split()
        vectors = [wv[w] for w in words if w in wv]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            from gensim.models import Word2Vec

            logger.info("Starting Word2Vec data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df["clean_text"] = train_df["text"].apply(clean_text)
            test_df["clean_text"] = test_df["text"].apply(clean_text)

            sentences = [t.split() for t in train_df["clean_text"]]

            logger.info("Training Word2Vec model (CBOW)...")
            w2v_model = Word2Vec(
                sentences,
                vector_size=self.vector_size,
                window=5,
                min_count=1,
                sg=0,       # CBOW
                epochs=15,
                workers=4,
            )

            save_object(self.config.model_path, w2v_model)
            logger.info(f"Word2Vec model saved. Vocab size: {len(w2v_model.wv)}")

            X_train = np.array([self._text_to_vector(t, w2v_model.wv) for t in train_df["clean_text"]])
            X_test  = np.array([self._text_to_vector(t, w2v_model.wv) for t in test_df["clean_text"]])
            y_train = train_df["label"].values
            y_test  = test_df["label"].values

            logger.info(f"Word2Vec feature matrix: {X_train.shape}")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            raise CustomException(e, sys)
