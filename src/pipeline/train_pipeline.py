import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logger.info("Step 2: Data Transformation")
            transformation = DataTransformation()
            X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Model Training
            logger.info("Step 3: Model Training")
            trainer = ModelTrainer()
            best_name, best_score, report = trainer.initiate_model_trainer(
                X_train, y_train, X_test, y_test
            )

            logger.info("=" * 60)
            logger.info(f"TRAINING COMPLETE | Best: {best_name} | F1: {best_score:.4f}")
            logger.info("=" * 60)

            return best_name, best_score, report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
