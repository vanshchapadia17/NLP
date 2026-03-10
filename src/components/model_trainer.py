import os
import sys
from dataclasses import dataclass

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self, model_path: str = None):
        self.config = ModelTrainerConfig()
        if model_path is not None:
            self.config.model_path = model_path

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, use_dense_nb: bool = False):
        try:
            logger.info("Starting model training")

            # MultinomialNB requires non-negative features (TF-IDF).
            # For Word2Vec dense vectors use GaussianNB instead.
            nb_model = GaussianNB() if use_dense_nb else MultinomialNB(alpha=0.1)
            nb_label = "Gaussian Naive Bayes" if use_dense_nb else "Multinomial Naive Bayes"

            models = {
                nb_label: nb_model,
                "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
                "Linear SVC": LinearSVC(C=1.0, max_iter=2000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models)

            # Pick best model by F1 score
            best_name = max(report, key=lambda k: report[k]["f1"])
            best_score = report[best_name]["f1"]

            logger.info(f"\n{'='*50}")
            logger.info("Model Evaluation Report:")
            for name, metrics in report.items():
                logger.info(
                    f"  {name}: Acc={metrics['accuracy']:.4f}  "
                    f"Prec={metrics['precision']:.4f}  "
                    f"Rec={metrics['recall']:.4f}  "
                    f"F1={metrics['f1']:.4f}"
                )
            logger.info(f"Best model: {best_name} (F1={best_score:.4f})")

            # Print to console as well
            print("\n" + "=" * 60)
            print("MODEL EVALUATION REPORT")
            print("=" * 60)
            for name, metrics in report.items():
                marker = " <<< BEST" if name == best_name else ""
                print(
                    f"  {name:<30} | Acc: {metrics['accuracy']:.4f} | "
                    f"Prec: {metrics['precision']:.4f} | "
                    f"Rec: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1']:.4f}{marker}"
                )
            print("=" * 60)

            # Re-train best model on full data and optionally save
            best_model = models[best_name]
            best_model.fit(X_train, y_train)
            if self.config.model_path:
                save_object(self.config.model_path, best_model)
                logger.info(f"Best model saved to {self.config.model_path}")

            return best_name, best_score, report

        except Exception as e:
            raise CustomException(e, sys)
