import os
import pickle
import numpy as np
from sklearn.svm import SVC

from src.models.base_model import BaseModel


class SVMModel(BaseModel):

    def __init__(self, config: dict) -> None:
        """
        Initialize an SVC using parameters from config.
        Read from config["model"]["svm"]:
        - C (default 1.0)
        - kernel (default 'rbf')
        Set probability=True so predict_proba() works.
        Store the classifier as self.model.
        """
        model_config = config["model"]["svm"]

        C = model_config.get("C", 1.0)
        kernel = model_config.get("kernel", "rbf")
        random_state = config["training"]["random_seed"]

        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)