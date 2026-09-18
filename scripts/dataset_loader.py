"""
dataset_loader.py

Loads the benchmark dataset and converts each row into
language-specific benchmark tasks.

Author: Shriya Patil
"""

from utils import load_config, load_dataset


class DatasetLoader:
    """
    Loads and prepares benchmark tasks.
    """

    def __init__(self):

        config = load_config()

        self.dataset_path = config["dataset"]["path"]

        self.df = load_dataset(self.dataset_path)

    def load_tasks(self):
        """
        Convert each dataset row into three benchmark tasks.

        Returns
        -------
        list[dict]
        """

        tasks = []

        language_columns = {
            "en": "prompt_en",
            "hi": "prompt_hi",
            "mr": "prompt_mr",
        }

        for _, row in self.df.iterrows():

            for language, column in language_columns.items():

                task = {

                    "attack_id": row["attack_id"],

                    "variation_id": row["variation_id"],

                    "attack_category": row["attack_category"],

                    "language": language,

                    "prompt": row[column],

                }

                tasks.append(task)

        return tasks
    
if __name__ == "__main__":

    loader = DatasetLoader()

    tasks = loader.load_tasks()

    print(f"Total Tasks: {len(tasks)}")

    print()

    print(tasks[0])