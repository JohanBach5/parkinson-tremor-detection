import os
import pickle
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, config: dict) -> None:
        """
        Initialize an XGBClassifier using parameters from config.
        Read from config["model"]["xgboost"]:
        - n_estimators (default 100)
        - max_depth (default 6)
        - learning_rate (default 0.1)
        - random_state from config["training"]["random_seed"]
        """
        model_config = config["model"]["xgboost"]
        task_type = config["training"].get("task_type", "classification")

        n_estimators = model_config.get("n_estimators", 100)
        max_depth = model_config.get("max_depth", 6)
        learning_rate = model_config.get("learning_rate", 0.1)
        random_state = config["training"]["random_seed"]

        if task_type == "classification":
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss"
            )
        elif task_type == "regression":
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                objective="reg:squarederror"
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)