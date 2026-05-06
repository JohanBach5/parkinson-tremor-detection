import os
import numpy as np

from src.models.factory import get_model
from src.evaluation.metrics import compute_all_metrics, aggregate_loso_metrics


class LOSOTrainer:

    def __init__(self, config: dict) -> None:
        """
        Store the config dictionary.
        Extract and store as instance variables:
        - model_type from config["training"]["model_type"]
        - random_seed from config["training"]["random_seed"]
        - models_dir from config["paths"]["models_dir"]
        - results_dir from config["paths"]["results_dir"]
        """
        self.config = config
        self.model_type = self.config["training"]["model_type"]
        self.random_seed = self.config["training"]["random_seed"]
        self.models_dir = self.config["paths"]["models_dir"]
        self.results_dir = self.config["paths"]["results_dir"]

    def run(
            self,
            windows: np.ndarray,
            labels: np.ndarray,
            subject_ids: np.ndarray
    ) -> dict:
        """
        Main LOSO training loop.
        Get all unique subject IDs with np.unique().
        For each unique subject:
            - call _get_fold_indices() to get train/test splits
            - extract X_train, y_train, X_test, y_test using indices
            - call _train_fold() and collect results
        Return dict:
        {
            "fold_results": list of per-fold metric dicts,
            "model_type": self.model_type
        }
        """
        unique_subjects = np.unique(subject_ids)
        fold_results = []

        for test_subject in unique_subjects:
            train_idx, test_idx = self._get_fold_indices(subject_ids, test_subject)

            X_train = windows[train_idx]
            y_train = labels[train_idx]
            X_test = windows[test_idx]
            y_test = labels[test_idx]

            result = self._train_fold(X_train, y_train, X_test, y_test, test_subject)
            fold_results.append(result)

        return {
            "fold_results": fold_results,
            "model_type": self.model_type
        }

    def _train_fold(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            fold_subject: str
    ) -> dict:
        """
        Train and evaluate one LOSO fold.
        Steps:
        1. Call get_model() to create a fresh model instance
        2. Call _handle_class_imbalance() on training data
        3. Call model.fit() on resampled training data
        4. Call model.predict() and model.predict_proba() on test data
        5. Call compute_all_metrics() on predictions
        6. Save model using model.save() to models_dir/fold_subject.pkl
        7. Add fold_subject to metrics dict
        Return the metrics dict.
        """
        model = get_model(self.model_type, self.config)
        X_train, y_train = self._handle_class_imbalance(X_train, y_train)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        model_path = os.path.join(self.models_dir, f"{fold_subject}.pkl")
        model.save(model_path)

        return {
            "metrics": metrics,
            "fold_subject": fold_subject,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

    @staticmethod
    def _handle_class_imbalance(
            X: np.ndarray,
            y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using random oversampling of minority class.
        Count samples in each class.
        Randomly duplicate minority class samples until classes are balanced.
        Use np.random.choice() with replace=True to oversample.
        Return resampled (X, y) with balanced classes.
        """
        minority_class = 1
        majority_class = 0

        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]

        n_to_sample = len(majority_idx) - len(minority_idx)

        if n_to_sample <= 0:
            return X, y

        oversampled_idx = np.random.choice(
            minority_idx,
            size=n_to_sample,
            replace=True
        )

        # combine original data with oversampled minority
        X_resampled = np.concatenate([X, X[oversampled_idx]])
        y_resampled = np.concatenate([y, y[oversampled_idx]])

        return X_resampled, y_resampled

    @staticmethod
    def _get_fold_indices(
            subject_ids: np.ndarray,
            test_subject: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return two boolean index arrays:
        - test_idx: True for all windows FROM test_subject
        - train_idx: True for all windows NOT from test_subject
        Use subject_ids == test_subject to build test_idx.
        train_idx is the logical NOT of test_idx using ~.
        """
        test_idx = np.array(subject_ids == test_subject)
        train_idx = ~test_idx

        return train_idx, test_idx


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    """
    Compute mean and std of each metric across all folds.
    Filter out non-numeric values before aggregating
    (e.g. fold_subject is a string — skip it).
    Call aggregate_loso_metrics() on the filtered results.
    Return the aggregated metrics dict.
    """
    metrics_only = [result["metrics"] for result in fold_results]
    return aggregate_loso_metrics(metrics_only)
