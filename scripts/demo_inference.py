import yaml
import pickle
import os
import numpy as np

from src.data.loader import get_loader
from src.data.preprocessor import preprocess_all_subjects
from src.inference.predictor import TremorPredictor


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------------
    config = load_config("configs/daphnet_config.yaml")
    print("✅ Config loaded")

    # -------------------------------------------------------------------------
    # 2. Pick a test subject and a trained model
    # We use S01R01 model to predict on S02R01 signal
    # (simulating a new unseen patient)
    # -------------------------------------------------------------------------
    test_subject = "S02R01"
    trained_on = "S01R01"
    model_path = os.path.join(config["paths"]["models_dir"], f"{trained_on}.pkl")

    print(f"\n🤖 Loading model trained on {trained_on}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"   Model loaded from {model_path}")

    # -------------------------------------------------------------------------
    # 3. Load the test subject's raw signal
    # -------------------------------------------------------------------------
    print(f"\n📂 Loading signal for {test_subject}...")
    loader = get_loader("daphnet", config)
    raw_data = loader.load_all_subjects()

    if test_subject not in raw_data:
        raise ValueError(f"Subject {test_subject} not found in dataset")

    df = raw_data[test_subject]
    print(f"   Signal loaded — {len(df)} samples")
    print(f"   Duration: {len(df) / config['sampling']['target_fs']:.1f} seconds")
    print(f"   True label distribution: {df['label'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # 4. Run inference
    # -------------------------------------------------------------------------
    print(f"\n🔍 Running inference...")

    # wrap model in a simple adapter since we saved the sklearn model directly
    class ModelAdapter:
        def __init__(self, sklearn_model):
            self.sklearn_model = sklearn_model

        def predict(self, X):
            return self.sklearn_model.predict(X)

        def predict_proba(self, X):
            return self.sklearn_model.predict_proba(X)

    adapted_model = ModelAdapter(model)
    predictor = TremorPredictor(adapted_model, config)
    results = predictor.predict_from_array(df)

    # -------------------------------------------------------------------------
    # 5. Print results
    # -------------------------------------------------------------------------
    print(f"\n📊 Inference Results for {test_subject}")
    print(f"   Total windows:     {results['n_windows']}")
    print(f"   FoG windows:       {results['fog_windows']}")
    print(f"   Normal windows:    {results['n_windows'] - results['fog_windows']}")
    print(f"   FoG rate:          {100 * results['fog_windows'] / results['n_windows']:.1f}%")

    # -------------------------------------------------------------------------
    # 6. Print time-annotated episodes
    # -------------------------------------------------------------------------
    window_size_sec = config["segmentation"]["window_size_sec"]
    step_size_sec = window_size_sec * (1 - config["segmentation"]["overlap"])

    predictions = results["predictions"]
    probabilities = results["probabilities"]

    print(f"\n🕐 Detected FoG episodes:")
    in_episode = False
    episode_start = 0

    for i, pred in enumerate(predictions):
        time_sec = i * step_size_sec
        if pred == 1 and not in_episode:
            in_episode = True
            episode_start = time_sec
        elif pred == 0 and in_episode:
            in_episode = False
            confidence = float(np.mean(probabilities[
                                       int(episode_start / step_size_sec):i, 1
                                       ]))
            print(f"   {episode_start:6.1f}s – {time_sec:6.1f}s "
                  f"(duration: {time_sec - episode_start:.1f}s, "
                  f"confidence: {confidence:.2f})")

    if in_episode:
        time_sec = len(predictions) * step_size_sec
        print(f"   {episode_start:6.1f}s – {time_sec:6.1f}s "
              f"(duration: {time_sec - episode_start:.1f}s)")

    if results["fog_windows"] == 0:
        print(f"   No FoG episodes detected")


if __name__ == "__main__":
    main()
