import os
import glob
import pandas as pd

COLUMN_NAMES = [
    "timestamp",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "hip_acc_x", "hip_acc_y", "hip_acc_z",
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "label"
]


class DaphnetLoader:

    def __init__(self, raw_data_dir: str, config: dict) -> None:
        """
        Store the path to raw data directory and config.
        Build the list of all .txt file paths found in raw_data_dir
        using glob. Raise an error if no files are found.
        """
        self.raw_data_dir = raw_data_dir
        self.config = config

        self.file_paths = glob.glob(os.path.join(self.raw_data_dir, "*.txt"))

        if not self.file_paths:
            raise FileNotFoundError(
                f"No .txt files found in {self.raw_data_dir}. "
                f"Please check the path in your config."
            )

    def load_all_subjects(self) -> dict[str, pd.DataFrame]:
        """
        Loop over every .txt file path found in __init__.
        For each file, extract the session key (e.g. 'S01R01')
        from the filename and call load_subject().
        Return a dictionary {session_key: DataFrame}.
        """
        data = {}

        for filepath in self.file_paths:
            filename = os.path.basename(filepath)
            session_key = os.path.splitext(filename)[0]
            data[session_key] = self.load_subject(filepath)

        return data

    def load_subject(self, filepath: str) -> pd.DataFrame:
        """
        Load one .txt file into a clean DataFrame by calling
        _parse_raw_file(), then _drop_unannotated_rows(), then
        _attach_metadata() in that exact order.
        Return the resulting DataFrame.
        """
        filename = os.path.basename(filepath)
        session_key = os.path.splitext(filename)[0]
        subject_id = session_key[:3]
        session_id = session_key[3:]

        df = self._parse_raw_file(filepath)
        df = self._drop_unannotated_rows(df)
        df = self._attach_metadata(df, subject_id, session_id)

        return df

    def save_processed(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Save each DataFrame in the data dictionary as a .csv file
        in the processed_data_dir defined in config.
        Create the directory if it does not exist using
        os.makedirs with exist_ok=True.
        Filename format: {session_key}.csv e.g. 'S01R01.csv'.
        """
        output_dir = self.config["paths"]["processed_data_dir"]

        os.makedirs(output_dir, exist_ok=True)

        for session_key, df in data.items():
            filepath = os.path.join(output_dir, f"{session_key}.csv")
            df.to_csv(filepath, index=False)

    @staticmethod
    def _parse_raw_file(filepath: str) -> pd.DataFrame:
        """
        Read a space-separated .txt file using pd.read_csv with
        sep='\s+' and header=None. Assign COLUMN_NAMES to the
        resulting DataFrame columns. Return the raw DataFrame.
        """
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        df.columns = COLUMN_NAMES

        return df

    @staticmethod
    def _drop_unannotated_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all rows where label == 0 as these are not part
        of the experiment. Reset the index after dropping.
        Return the filtered DataFrame.
        """
        df = df[df["label"] != 0]
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _attach_metadata(
            df: pd.DataFrame,
            subject_id: str,
            session_id: str
    ) -> pd.DataFrame:
        """
        Add two new columns to the DataFrame:
        - subject_id: e.g. 'S01' extracted from filename
        - session_id: e.g. 'R01' extracted from filename
        Return the DataFrame with these two new columns.
        """
        df["subject_id"] = subject_id
        df["session_id"] = session_id

        return df


def get_loader(dataset_name: str, config: dict) -> DaphnetLoader:
    """
    Factory function that takes a dataset name string and config.
    If dataset_name == 'daphnet' return a DaphnetLoader instance.
    Raise a ValueError for any unrecognised dataset name.
    """
    if dataset_name == "daphnet":
        raw_data_dir = config["paths"]["raw_data_dir"]
        return DaphnetLoader(raw_data_dir, config)
    else:
        raise ValueError(
            f"Unrecognised dataset name: {dataset_name}"
        )
