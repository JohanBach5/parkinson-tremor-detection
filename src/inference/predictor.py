import numpy as np
import pandas as pd
from src.data.preprocessor import SignalPreprocessor
from src.data.segmentor import SlidingWindowSegmentor
from src.features.feature_pipeline import FeaturePipeline
from src.models.base_model import BaseModel
from src.inference.postprocessor import postprocess


class TremorPredictor:

    def __init__(self, model: BaseModel, config: dict) -> None:
        """
        Store the loaded model and config.
        Instantiate preprocessor, segmentor, and feature pipeline.
        """
        self.model = model
        self.config = config
        self.preprocessor = SignalPreprocessor(config)
        self.segmentor = SlidingWindowSegmentor(config)
        self.pipeline = FeaturePipeline(config)

    def predict_from_array(self, df: pd.DataFrame) -> dict:
        """
        Run full pipeline on a raw DataFrame.
        Returns predictions, probabilities and summary stats.
        """
        df = self._run_preprocessing(df)
        windows, _ = self._run_segmentation(df)
        X = self._run_feature_extraction(windows)
        predictions, probabilities = self._run_model(X)
        predictions, probabilities = postprocess(predictions, probabilities)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "n_windows": len(predictions),
            "fog_windows": int(np.sum(predictions == 1))
        }

    def _run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run SignalPreprocessor.process_subject() on the input DataFrame.
        """
        return self.preprocessor.process_subject(df)

    def _run_segmentation(self, df: pd.DataFrame) -> tuple:
        """
        Run SlidingWindowSegmentor.segment_subject() on the DataFrame.
        """
        return self.segmentor.segment_subject(df)

    def _run_feature_extraction(self, windows: np.ndarray) -> np.ndarray:
        """
        Run FeaturePipeline.extract() on the windows.
        """
        return self.pipeline.extract(windows)

    def _run_model(
            self,
            X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run model.predict() and model.predict_proba().
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
