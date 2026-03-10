import os
import sys
import urllib.request
import zipfile
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "spam_raw.csv")
    train_data_path: str = os.path.join("artifacts", "spam_train.csv")
    test_data_path: str = os.path.join("artifacts", "spam_test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def _download_dataset(self) -> pd.DataFrame:
        """Download SMS Spam Collection dataset from UCI repository."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = os.path.join("artifacts", "smsspam.zip")
        os.makedirs("artifacts", exist_ok=True)

        logger.info("Downloading SMS Spam Collection dataset...")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("artifacts")

        tsv_path = os.path.join("artifacts", "SMSSpamCollection")
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["label", "text"],
                         encoding="latin-1")
        os.remove(zip_path)
        logger.info(f"Downloaded dataset: {df.shape[0]} rows")
        return df

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion")
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Load from local if already downloaded
            tsv_path = os.path.join("artifacts", "SMSSpamCollection")
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep="\t", header=None,
                                 names=["label", "text"], encoding="latin-1")
                logger.info("Loaded dataset from local cache")
            else:
                df = self._download_dataset()

            # Map labels to binary
            df["label"] = df["label"].map({"ham": 0, "spam": 1})
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)

            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved: {df.shape}")

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                                  stratify=df["label"])
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(f"Train: {train_df.shape} | Test: {test_df.shape}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
