from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger

from ..baselines import BASELINE_MAP
from ..mdad_datasets import DATASET_MAP
from . import CrossLanguageExperimentConfig, ExperimentConfig

__all__ = ["ExperimentRunner"]


class ExperimentRunner:
    def __init__(self, data_dir: str, baselines: List[str], device: str = "cuda"):
        self.data_dir = data_dir
        self.baseline_names = baselines if isinstance(baselines, list) else [baselines]
        self.device = device
        self.results: dict[str, Any] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dataset_cache: Dict[Tuple[str, str | None], Any] = {}

        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        log_file = f"logs/experiment_{self.timestamp}.log"
        logger.add(log_file, rotation="20 MB", retention="30 days")

    def _create_combined_dataset(
        self, dataset_names: List[str], split: str, subset: str | None = None, limit: int | None = None
    ) -> Tuple[List[Any], List[Any]]:
        """Combine multiple datasets for a specific split, using a cache."""

        def _get_dataset(name: str, subset: str | None):
            key = (name, subset)
            if key not in self._dataset_cache:
                kwargs = {"data_dir": self.data_dir}
                if name == "phonecall" and subset is not None:
                    kwargs["subset"] = subset
                self._dataset_cache[key] = DATASET_MAP[name](**kwargs)
            return self._dataset_cache[key]

        combined_data: List[Any] = []
        combined_labels: List[Any] = []

        for ds_name in dataset_names:
            ds = _get_dataset(ds_name, subset if ds_name == "phonecall" else None)
            split_data = ds.data.get(split, [])
            if not split_data:
                continue

            combined_data.extend(split_data)
            combined_labels.extend(ds.labels[split])
            logger.info(f"Added {len(split_data)} samples from {ds_name} {split}")

        if limit is not None and len(combined_data) > limit:
            torch.manual_seed(34)
            idx = torch.randperm(len(combined_data))[:limit].tolist()
            combined_data = [combined_data[i] for i in idx]
            combined_labels = [combined_labels[i] for i in idx]

        logger.info(f"Combined {split}: {len(combined_data)} total samples from {len(dataset_names)} datasets")
        return combined_data, combined_labels

    def _train_model(
        self, train_datasets: List[str], val_datasets: List[str], expr_name: str, subset: str | None, baseline_name: str
    ) -> Any:
        """Train a model on specified datasets."""
        logger.info(f"Starting training for experiment {expr_name}")
        logger.info(f"Training {baseline_name} on {train_datasets}")

        train_data, train_labels = self._create_combined_dataset(train_datasets, "train", subset)
        val_data, val_labels = self._create_combined_dataset(val_datasets, "dev", subset)

        if not train_data:
            logger.warning(f"No training data found for {train_datasets}")
            return None

        baseline = BASELINE_MAP[baseline_name](device=self.device)
        ref_data, ref_labels = None, None
        if baseline.name == "MKRT":
            ref_num = baseline.ref_num
            if len(train_data) >= ref_num:
                ref_data, ref_labels = train_data[:ref_num], train_labels[:ref_num]
                train_data, train_labels = train_data[ref_num:], train_labels[ref_num:]
                logger.info(f"Extracted {ref_num} reference samples for MKRT")
            else:
                logger.warning(f"Not enough training samples for MKRT reference (need {ref_num}, got {len(train_data)})")

        baseline.train(
            train_data=train_data,
            train_labels=train_labels,
            eval_data=val_data,
            eval_labels=val_labels,
            ref_data=ref_data,
            ref_labels=ref_labels,
            dataset_name=expr_name,
            sr=16000,
        )
        return baseline

    def _evaluate_model(
        self, baseline: Any, test_datasets: List[str], expr_name: str, subset: str | None, baseline_name: str
    ) -> Dict[str, float]:
        """Evaluate a trained model on test datasets."""
        if baseline is None:
            return {"eer": float("inf")}

        logger.info(f"Start evaluation for experiment {expr_name}")
        logger.info(f"Evaluating {baseline_name} on {test_datasets}")

        if len(test_datasets) == 1 and test_datasets[0] in ["partialfake", "noisyspeech"]:
            dataset = DATASET_MAP[test_datasets[0]](data_dir=self.data_dir)
            return dataset.evaluate(baseline, ["eer"], in_domain=True, expr_name=expr_name)

        test_data, test_labels = self._create_combined_dataset(test_datasets, "test", subset)
        if not test_data:
            logger.warning(f"No test data found for {test_datasets}")
            return {"eer": float("inf")}

        return baseline.evaluate(
            data=test_data, labels=test_labels, metrics=["eer"], sr=16000, in_domain=True, dataset_name=expr_name
        )

    def _run_experiment(self, name: str, config: ExperimentConfig) -> dict[str, Any]:
        """Run a single experiment described by *config* for all baselines."""
        logger.info(f"Running {name} with baselines: {', '.join(self.baseline_names)}")
        experiment_results: dict[str, Any] = {}

        for baseline_name in self.baseline_names:
            logger.info(f"  -> Baseline: {baseline_name}")
            model = self._train_model(
                train_datasets=config.train_datasets,
                val_datasets=config.val_datasets,
                expr_name=name,
                subset=config.subset,
                baseline_name=baseline_name,
            )

            baseline_results: dict[str, Any] = {
                test_name: self._evaluate_model(
                    baseline=model,
                    test_datasets=datasets,
                    expr_name=name,
                    subset=config.subset,
                    baseline_name=baseline_name,
                )
                for test_name, datasets in config.test_sets.items()
            }
            experiment_results[baseline_name] = baseline_results

        self.results[name] = experiment_results
        self.save_results()
        return experiment_results

    def _run_cross_language_experiment(self, name: str, config: CrossLanguageExperimentConfig) -> Dict[str, Any]:
        """Run an experiment that trains separate models per language."""
        logger.info(f"Running cross-language experiment {name}")
        exp_results: dict[str, Any] = {}

        for baseline_name in self.baseline_names:
            logger.info(f"  -> Baseline: {baseline_name}")

            language_models = {
                lang_cfg.name: self._train_model(
                    train_datasets=lang_cfg.train_datasets,
                    val_datasets=lang_cfg.val_datasets,
                    expr_name=f"{name}_{lang_cfg.name}",
                    subset=lang_cfg.subset,
                    baseline_name=baseline_name,
                )
                for lang_cfg in config.languages
            }

            baseline_results: dict[str, Any] = {}
            for model_lang, model in language_models.items():
                results_for_model = {
                    test_lang: self._evaluate_model(
                        baseline=model,
                        test_datasets=datasets,
                        expr_name=f"{name}_{model_lang}",
                        subset=next((lc.subset for lc in config.languages if lc.name == test_lang), None),
                        baseline_name=baseline_name,
                    )
                    for test_lang, datasets in config.test_sets.items()
                }
                baseline_results[f"model_{model_lang}"] = results_for_model
            exp_results[baseline_name] = baseline_results

        self.results[name] = exp_results
        self.save_results()
        return exp_results

    def save_results(self):
        """Save results to a JSON file."""
        results_file = f"results/result_{self.timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
