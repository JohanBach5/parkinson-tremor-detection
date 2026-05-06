from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ) -> None:
        """
        Train the model on the given feature matrix and labels.
        Input:
        - X_train shape: [n_samples, n_features]
        - y_train shape: [n_samples]
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for the given feature matrix.
        Input shape: [n_samples, n_features]
        Return shape: [n_samples] — binary labels (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given feature matrix.
        Input shape: [n_samples, n_features]
        Return shape: [n_samples, n_classes]
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk at the given path.
        Create the output directory if it does not exist.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from disk at the given path.
        """
        pass

    def get_model_name(self) -> str:
        """
        Return the class name as a string.
        Use self.__class__.__name__.
        This is not abstract — it works for all subclasses automatically.
        """
        return self.__class__.__name__
