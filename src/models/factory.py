from src.models.base_model import BaseModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel
from src.models.xgboost_model import XGBoostModel


def get_model(model_name: str, config: dict) -> BaseModel:
    """
    Factory function that returns the right model instance
    based on model_name string.
    Raise ValueError for unrecognised model names.
    """
    if model_name == "random_forest":
        return RandomForestModel(config)
    elif model_name == "svm":
        return SVMModel(config)
    elif model_name == "xgboost":
        return XGBoostModel(config)
    else:
        raise ValueError(
            f"Unrecognised model name: {model_name}. "
            f"Options are: 'random_forest', 'svm', 'xgboost'."
        )
