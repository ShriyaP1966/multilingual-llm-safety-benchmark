"""
result_writer.py

Writes benchmark responses to CSV files.

Author: Shriya Patil
"""

import pandas as pd
from pathlib import Path

from utils import (
    load_config,
    ensure_directory,
    save_dataframe,
)


class ResultWriter:
    """
    Saves benchmark results.
    """

    def __init__(self):

        config = load_config()

        self.output_directory = config["outputs"]["raw"]

        ensure_directory(
            self.output_directory
        )

    def save_results(
        self,
        responses,
        filename="results.csv"
    ):
        """
        Save a complete list of responses to CSV.
        """

        df = pd.DataFrame(responses)

        output_path = (
            Path(self.output_directory)
            / filename
        )

        save_dataframe(
            df,
            str(output_path)
        )

        print()
        print("Results saved successfully.")

        return output_path

    def append_result(
        self,
        result,
        filename="results.csv"
    ):
        """
        Append a single result to a CSV file.

        Creates the file if it does not already exist.
        """

        output_path = (
            Path(self.output_directory)
            / filename
        )

        result_df = pd.DataFrame(
            [result]
        )

        file_exists = output_path.exists()

        result_df.to_csv(
            output_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig"
        )

        return output_path