import os

import pandas as pd


class CISPDLoader:

    def __init__(self, raw_data_dir: str, labels_path: str, config: dict) -> None:
        """
        Store raw_data_dir, labels_path and config.
        raw_data_dir points to:
        data/cis_pd/sensor_data/cis_pd_sensor_data/smartwatch_accelerometer/
        labels_path points to:
        data/cis_pd/labels/CIS-PD_Training_Data_IDs_Labels.csv

        Load the labels CSV into self.labels as a DataFrame.
        Extract valid_only from config["dataset"]["valid_only"].
        If valid_only is True, drop rows where tremor label is NaN.
        Store the list of valid measurement_ids as self.measurement_ids.
        Raise FileNotFoundError if raw_data_dir does not exist.
        """
        self.raw_data_dir = raw_data_dir
        self.labels_path = labels_path
        self.config = config
        self.valid_only = self.config["dataset"]["valid_only"]

        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(
                f"Raw data directory not found: {raw_data_dir}"
            )

        self.labels = pd.read_csv(labels_path)

        if self.valid_only:
            self.labels = self.labels.dropna(subset=["tremor"])

        self.measurement_ids = self.labels["measurement_id"].tolist()

    def load_all_subjects(self) -> dict[str, pd.DataFrame]:
        """
        Loop over self.measurement_ids.
        For each measurement_id call load_subject().
        Return dictionary {measurement_id: DataFrame}.
        Skip measurements where the sensor file does not exist
        and print a warning.
        """
        data = {}

        for measurement_id in self.measurement_ids:
            filepath = os.path.join(self.raw_data_dir, f"{measurement_id}.csv")

            if not os.path.exists(filepath):
                print(f"[WARNING] Missing sensor file: {measurement_id}")
                continue

            data[measurement_id] = self.load_subject(measurement_id)

        return data

    def load_subject(self, measurement_id: str) -> pd.DataFrame:
        """
        Load one CSV file for the given measurement_id.
        File path: raw_data_dir / measurement_id.csv
        Columns in file: Timestamp, X, Y, Z
        Rename columns to: timestamp, wrist_acc_x, wrist_acc_y, wrist_acc_z
        Attach metadata by calling _attach_metadata().
        Return the DataFrame.
        """
        filepath = os.path.join(self.raw_data_dir, f"{measurement_id}.csv")

        df = pd.read_csv(filepath)
        df.columns = ["timestamp", "wrist_acc_x", "wrist_acc_y", "wrist_acc_z"]

        # get subject_id and tremor_score from labels
        row = self.labels[self.labels["measurement_id"] == measurement_id].iloc[0]
        subject_id = str(row["subject_id"])
        tremor_score = float(row["tremor"])

        df = self._attach_metadata(df, measurement_id, subject_id, tremor_score)

        return df

    def get_labels(self) -> pd.DataFrame:
        """
        Return self.labels DataFrame containing:
        measurement_id, subject_id, tremor (and other UPDRS columns)
        """
        return self.labels

    def save_processed(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Save each DataFrame as a .csv file in
        config["paths"]["processed_data_dir"].
        Create directory if it does not exist.
        Filename format: {measurement_id}.csv
        """
        output_dir = self.config["paths"]["processed_data_dir"]
        os.makedirs(output_dir, exist_ok=True)

        for measurement_id, df in data.items():
            filepath = os.path.join(output_dir, f"{measurement_id}.csv")
            df.to_csv(filepath, index=False)

    @staticmethod
    def _attach_metadata(
            df: pd.DataFrame,
            measurement_id: str,
            subject_id: str,
            tremor_score: float
    ) -> pd.DataFrame:
        """
        Add three new columns to the DataFrame:
        - measurement_id: the measurement ID string
        - subject_id: the subject ID string
        - tremor_score: the UPDRS tremor label (float, can be 0.0-4.0)
        Return the DataFrame with these columns.
        """
        df["measurement_id"] = measurement_id
        df["subject_id"] = subject_id
        df["tremor_score"] = tremor_score

        return df
