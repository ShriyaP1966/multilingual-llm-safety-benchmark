"""
run_experiment.py

Main experiment runner for the Multilingual LLM Safety Benchmark.

This script:
1. Loads configuration
2. Validates the dataset
3. Creates a new experiment directory
4. Initializes the selected provider
5. Runs the benchmark
6. Saves model responses
"""

from pathlib import Path
from datetime import datetime
import json
import argparse
import pandas as pd

from utils import (
    load_config,
    setup_logger,
)

from validate_dataset import validate_dataset
from dataset_loader import DatasetLoader
from result_writer import ResultWriter

from providers.provider_factory import ProviderFactory


class ExperimentRunner:
    """
    Main experiment controller.
    """

    def __init__(self):

        self.config = load_config()

        self.logger = setup_logger(
            logger_name="experiment"
        )

        self.provider = ProviderFactory.create_provider(
            self.config
        )

        self.experiment_path = None
        self.limit = None

    def create_experiment_directory(self):
        """
        Create a unique experiment folder.
        """

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.experiment_path = (
            Path(self.config["experiments"]["directory"])
            / f"EXP_{timestamp}"
        )

        self.experiment_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Experiment Folder:")
        print(self.experiment_path)

        self.logger.info(
            f"Experiment Folder: {self.experiment_path}"
        )

    def model_name(self):
        """
        Model identifier for the active provider.
        """

        return self.config["models"][
            self.config["provider"]
        ]["model"]


    def output_path(self):
        """
        Path of the raw results file for the active model.

        run() and execute_tasks() must agree on this, so both
        derive it here rather than rebuilding it separately.
        """

        safe_model_name = (
            self.model_name()
            .replace("/", "_")
            .replace("-", "_")
        )

        return (
            Path(self.config["outputs"]["raw"])
            / f"{safe_model_name}_results.csv"
        )


    def save_metadata(self, status="initialized", **extra):
        """
        Save experiment metadata.

        Called once before execution with the default status and
        again on completion, so a finished run is distinguishable
        from one that died partway through. Previously every
        experiment stayed at "initialized" forever and recorded
        nothing about its results.
        """

        metadata = {

            "experiment_time": datetime.now().isoformat(),

            "model": self.model_name(),

            "provider": self.config["provider"],

            "dataset": self.config["dataset"]["path"],

            "generation": self.config["generation"],

            "status": status

        }

        metadata.update(extra)

        with open(
            self.experiment_path / "metadata.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )


    def execute_tasks(self, tasks):
        """
        Execute benchmark tasks using the selected provider.

        Already completed tasks are skipped. Each new response
        is saved immediately when save_every_response is enabled.
        """

        responses = []

        total = len(tasks)

        writer = ResultWriter()

        output_path = self.output_path()

        output_filename = output_path.name

        completed_keys = set()

        if output_path.exists():
            existing_df = pd.read_csv(
                output_path,
                encoding="utf-8",
                keep_default_na=False
            )

            for _, row in existing_df.iterrows():

                response_text = str(row.get("response", "")).strip()

                if response_text:
                    key = (
                        str(row["attack_id"]),
                        str(row["variation_id"]),
                        str(row["language"])
                    )

                    completed_keys.add(key)

            print(
                f"Found {len(completed_keys)} "
                f"already-completed responses."
            )

        skipped = 0

        print()
        print("Running benchmark...")
        print()

        for index, task in enumerate(
            tasks,
            start=1
        ):

            task_key = (
                str(task["attack_id"]),
                str(task["variation_id"]),
                str(task["language"])
            )

            if task_key in completed_keys:

                skipped += 1

                print(
                    f"[{index}/{total}] Skipped "
                    f"(already completed)"
                )

                continue

            response = self.provider.generate_response(
                task["prompt"]
            )
            response_text = response["text"]
            input_tokens = response["input_tokens"]
            output_tokens = response["output_tokens"]
            total_tokens = response["total_tokens"]            

            task_result = {
                "attack_id": task["attack_id"],
                "variation_id": task["variation_id"],
                "attack_category": task["attack_category"],
                "language": task["language"],
                "prompt": task["prompt"],
                "model": self.config["models"][
                    self.config["provider"]
                ]["model"],
                "provider": self.config["provider"],
                "response": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

            responses.append(task_result)

            if self.config["runtime"]["save_every_response"]:

                writer.append_result(
                    task_result,
                    filename=output_filename
                )

            completed_keys.add(task_key)

            print(
                f"[{index}/{total}] Completed"
            )

        print()
        print(
            f"Skipped: {skipped} already-completed tasks."
        )

        print(
            f"New responses generated: {len(responses)}"
        )

        return responses

    def run(self, limit=None):

        print("=" * 60)
        print("MULTILINGUAL SAFETY BENCHMARK")
        print("=" * 60)

        print("\nValidating dataset...\n")

        if not validate_dataset():

            print("\nDataset validation failed.")

            return

        print("\nDataset validation passed.\n")

        loader = DatasetLoader()

        tasks = loader.load_tasks()

        if limit is not None:
            if limit <= 0:
                raise ValueError(
                    "Limit must be greater than 0."
                )

            tasks = tasks[:limit]

        print(
            f"Loaded {len(tasks)} benchmark tasks."
        )

        # Create experiment directory before
        # saving metadata and results.
        self.create_experiment_directory()

        self.save_metadata()

        responses = self.execute_tasks(
            tasks
        )

        print()

        print(
            f"Generated {len(responses)} responses."
        )

        output_path = self.output_path()

        self.save_metadata(
            status="completed",
            benchmark_tasks=len(tasks),
            responses_created=len(responses),
            output_file=str(output_path),
        )

        print()

        print("=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)

        print(
            f"Provider          : "
            f"{self.config['provider']}"
        )

        print(
            f"Model             : "
            f"{self.config['models'][self.config['provider']]['model']}"
        )

        print(
            f"Benchmark Tasks   : "
            f"{len(tasks)}"
        )

        print(
            f"Responses Created : "
            f"{len(responses)}"
        )

        print(
            f"Output File       : "
            f"{output_path}"
        )

        print(
            f"Experiment Folder : "
            f"{self.experiment_path}"
        )

        print()
        print("STATUS : SUCCESS")
        print("=" * 60)

        self.logger.info(
            "Experiment completed successfully."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the multilingual safety benchmark."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of benchmark tasks to run."
    )

    args = parser.parse_args()

    runner = ExperimentRunner()

    runner.run(limit=args.limit)