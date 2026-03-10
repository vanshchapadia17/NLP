import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, Word2VecTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Step 1: Data Ingestion (shared for both vectorizers)
            logger.info("Step 1: Data Ingestion")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # ── Step 2a: TF-IDF ──────────────────────────────────────
            logger.info("Step 2a: TF-IDF Transformation")
            tfidf_transform = DataTransformation()
            X_tr_tfidf, y_train, X_te_tfidf, y_test = tfidf_transform.initiate_data_transformation(
                train_path, test_path
            )

            logger.info("Step 3a: TF-IDF Model Training")
            tfidf_trainer = ModelTrainer()          # saves model.pkl
            tfidf_best, tfidf_f1, tfidf_report = tfidf_trainer.initiate_model_trainer(
                X_tr_tfidf, y_train, X_te_tfidf, y_test, use_dense_nb=False
            )

            # ── Step 2b: Word2Vec ─────────────────────────────────────
            logger.info("Step 2b: Word2Vec Transformation")
            w2v_transform = Word2VecTransformation()
            X_tr_w2v, y_train, X_te_w2v, y_test = w2v_transform.initiate_data_transformation(
                train_path, test_path
            )

            logger.info("Step 3b: Word2Vec Model Training")
            w2v_trainer = ModelTrainer(model_path="")    # empty string = skip saving, comparison only
            w2v_best, w2v_f1, w2v_report = w2v_trainer.initiate_model_trainer(
                X_tr_w2v, y_train, X_te_w2v, y_test, use_dense_nb=True
            )

            # ── Summary ───────────────────────────────────────────────
            logger.info("=" * 60)
            logger.info(f"TF-IDF   best: {tfidf_best} | F1: {tfidf_f1:.4f}")
            logger.info(f"Word2Vec best: {w2v_best}   | F1: {w2v_f1:.4f}")
            logger.info("Prediction will use TF-IDF model (generally stronger on small datasets)")
            logger.info("=" * 60)

            comparison = {
                "tfidf": {
                    "best_model": tfidf_best,
                    "f1_score": round(float(tfidf_f1), 4),
                    "report": tfidf_report,
                },
                "word2vec": {
                    "best_model": w2v_best,
                    "f1_score": round(float(w2v_f1), 4),
                    "report": w2v_report,
                },
            }

            return tfidf_best, tfidf_f1, tfidf_report, comparison

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
