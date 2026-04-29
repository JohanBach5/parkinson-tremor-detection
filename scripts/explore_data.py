import yaml
import numpy as np

from src.data.loader import get_loader
from src.data.preprocessor import preprocess_all_subjects
from src.data.segmentor import segment_all_subjects


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
    # 2. Load raw data
    # -------------------------------------------------------------------------
    loader = get_loader("daphnet", config)
    raw_data = loader.load_all_subjects()

    print(f"\n✅ Raw data loaded")
    print(f"   Number of sessions: {len(raw_data)}")

    for session_key, df in raw_data.items():
        print(f"   {session_key} → {len(df)} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # 3. Inspect one session
    # -------------------------------------------------------------------------
    first_key = list(raw_data.keys())[0]
    first_df = raw_data[first_key]

    print(f"\n✅ Sample session: {first_key}")
    print(f"   Columns: {list(first_df.columns)}")
    print(f"   Label distribution:")
    print(f"   {first_df['label'].value_counts().to_dict()}")
    print(f"\n   First 3 rows:")
    print(f"   {first_df.head(3)}")

    # -------------------------------------------------------------------------
    # 4. Save processed data
    # -------------------------------------------------------------------------
    loader.save_processed(raw_data)
    print(f"\n✅ Processed data saved to {config['paths']['processed_data_dir']}")

    # -------------------------------------------------------------------------
    # 5. Preprocess
    # -------------------------------------------------------------------------
    preprocessed_data = preprocess_all_subjects(raw_data, config)

    print(f"\n✅ Preprocessing complete")
    print(f"   Sessions preprocessed: {len(preprocessed_data)}")

    first_preprocessed = preprocessed_data[first_key]
    print(f"   Sample mean after normalization (should be ~0 for zscore):")
    print(f"   {first_preprocessed[['ankle_acc_x', 'hip_acc_x', 'wrist_acc_x']].mean().round(4).to_dict()}")

    # -------------------------------------------------------------------------
    # 6. Segment
    # -------------------------------------------------------------------------
    all_windows, all_labels, all_subject_ids = segment_all_subjects(
        preprocessed_data, config
    )

    print(f"\n✅ Segmentation complete")
    print(f"   Total windows:     {all_windows.shape[0]}")
    print(f"   Window shape:      {all_windows.shape[1]} samples × {all_windows.shape[2]} channels")
    print(f"   Labels shape:      {all_labels.shape}")
    print(f"   Subject IDs shape: {all_subject_ids.shape}")

    # -------------------------------------------------------------------------
    # 7. Class balance
    # -------------------------------------------------------------------------
    fog_count = np.sum(all_labels == 1)
    normal_count = np.sum(all_labels == 0)
    total = len(all_labels)

    print(f"\n✅ Class distribution")
    print(f"   Normal (0): {normal_count} windows ({100 * normal_count / total:.1f}%)")
    print(f"   FoG    (1): {fog_count} windows ({100 * fog_count / total:.1f}%)")

    # -------------------------------------------------------------------------
    # 8. Subject breakdown
    # -------------------------------------------------------------------------
    print(f"\n✅ Windows per subject")
    unique_subjects = np.unique(all_subject_ids)
    for subject in unique_subjects:
        mask = all_subject_ids == subject
        subject_fog = np.sum(all_labels[mask] == 1)
        subject_normal = np.sum(all_labels[mask] == 0)
        print(f"   {subject} → {np.sum(mask)} windows "
              f"(normal: {subject_normal}, FoG: {subject_fog})")


if __name__ == "__main__":
    main()