import sys
from src.exception import CustomException
from src.logger import logger
from src.utils import load_object
from src.components.data_transformation import clean_text


class PredictPipeline:
    def __init__(self):
        self.vectorizer = load_object("artifacts/tfidf_vectorizer.pkl")
        self.model = load_object("artifacts/model.pkl")

    def predict(self, texts: list) -> list:
        try:
            cleaned = [clean_text(t) for t in texts]
            features = self.vectorizer.transform(cleaned)
            predictions = self.model.predict(features)
            labels = ["spam" if p == 1 else "ham" for p in predictions]
            return labels
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, texts: list) -> list:
        """Return spam probability (if model supports it)."""
        try:
            cleaned = [clean_text(t) for t in texts]
            features = self.vectorizer.transform(cleaned)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)
                return [round(float(p[1]), 4) for p in probs]
            else:
                # For LinearSVC use decision function
                scores = self.model.decision_function(features)
                # Normalize to [0,1] roughly
                import numpy as np
                from scipy.special import expit
                return [round(float(expit(s)), 4) for s in scores]
        except Exception as e:
            raise CustomException(e, sys)
