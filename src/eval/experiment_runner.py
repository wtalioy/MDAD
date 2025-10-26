#!/usr/bin/env python3
"""
Experiment Runner for MDAD Benchmark Experiments

This script implements the four experiments described in experiment_plan.md:
1. Domain Generalization Stress Test (Scripted-to-Spontaneous)
2. Emotional Prosody Uncanny Valley Test
3. Sensitivity vs Robustness Test
4. Cross-Language Generalization Test
"""

import os
import argparse
import json
import warnings
from typing import Dict, List, Tuple, Any
from datetime import datetime
import torch
from loguru import logger

from baselines import BASELINE_MAP
from mdad_datasets import DATASET_MAP


class ExperimentRunner:
    def __init__(self, data_dir: str = "data/MDAD", baselines: List[str] = ["aasist"], device: str = "cuda"):
        self.data_dir = data_dir
        self.baseline_names = baselines if isinstance(baselines, list) else [baselines]
        self.device = device
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/experiments_{self.timestamp}.log"
        logger.add(log_file, rotation="20 MB", retention="30 days")
        
    def _create_combined_dataset(self, dataset_names: List[str], split: str, subset: str = None, limit: int = None) -> Tuple[List, List]:
        """Combine multiple datasets for a specific split"""
        combined_data = []
        combined_labels = []
        
        for dataset_name in dataset_names:
            # Handle PhoneCall subset
            if dataset_name == "phonecall" and subset:
                dataset = DATASET_MAP[dataset_name](data_dir=self.data_dir, subset=subset)
            else:
                dataset = DATASET_MAP[dataset_name](data_dir=self.data_dir)
            
            if split in dataset.data and len(dataset.data[split]) > 0:
                combined_data.extend(dataset.data[split])
                combined_labels.extend(dataset.labels[split])
                logger.info(f"Added {len(dataset.data[split])} samples from {dataset_name} {split}")
        
        if limit is not None and len(combined_data) > limit:
            torch.manual_seed(34)
            indices = torch.randperm(len(combined_data))[:limit].tolist()
            combined_data = [combined_data[i] for i in indices]
            combined_labels = [combined_labels[i] for i in indices]
        
        logger.info(f"Combined {split}: {len(combined_data)} total samples")
        return combined_data, combined_labels
    
    def _train_model(self, train_datasets: List[str], val_datasets: List[str], expr_name: str, subset: str = None, baseline_name: str = None) -> Any:
        """Train a model on specified datasets"""
        baseline_name = baseline_name or self.baseline_names[0]
        logger.info(f"Starting training for experiment {expr_name}")
        logger.info(f"Training {baseline_name} on {train_datasets}")
        
        # Create combined training data
        train_data, train_labels = self._create_combined_dataset(train_datasets, 'train', subset)
        val_data, val_labels = self._create_combined_dataset(val_datasets, 'dev', subset)
        
        if len(train_data) == 0:
            logger.warning(f"No training data found for {train_datasets }")
            return None
            
        # Create baseline model
        baseline = BASELINE_MAP[baseline_name](device=self.device)
        
        # Handle ref_data and ref_labels for MKRT
        ref_data = None
        ref_labels = None
        if baseline.name == 'MKRT':
            # For MKRT, extract reference data from the first ref_num samples
            ref_num = baseline.ref_num
            if len(train_data) >= ref_num:
                ref_data = train_data[:ref_num]
                ref_labels = train_labels[:ref_num]
                train_data = train_data[ref_num:]
                train_labels = train_labels[ref_num:]
                logger.info(f"Extracted {ref_num} reference samples for MKRT")
            else:
                logger.warning(f"Not enough training samples for MKRT reference (need {ref_num}, got {len(train_data)})")
        
        # Train the model
        baseline.train(
            train_data=train_data,
            train_labels=train_labels,
            eval_data=val_data,
            eval_labels=val_labels,
            ref_data=ref_data,
            ref_labels=ref_labels,
            dataset_name=expr_name,
            sr=16000
        )
        
        return baseline
    
    def _evaluate_model(self, baseline: Any, test_datasets: List[str], expr_name: str, subset: str = None, baseline_name: str = None) -> Dict[str, float]:
        """Evaluate a trained model on test datasets"""
        baseline_name = baseline_name or self.baseline_names[0]
        if baseline is None:
            return {"eer": float('inf')}
            
        logger.info(f"Start evaluation for experiment {expr_name}")
        logger.info(f"Evaluating {baseline_name} on {test_datasets}")
        
        # Handle special datasets that have their own evaluation methods
        if len(test_datasets) == 1 and test_datasets[0] in ['partialfake', 'noisyspeech']:
            dataset_name = test_datasets[0]
            dataset = DATASET_MAP[dataset_name](data_dir=self.data_dir)
            results = dataset.evaluate(baseline, ["eer"], in_domain=True, expr_name=expr_name)
            return results
        
        # Create combined test data for regular datasets
        test_data, test_labels = self._create_combined_dataset(test_datasets, 'test', subset)
        
        if len(test_data) == 0:
            logger.warning(f"No test data found for {test_datasets}")
            return {"eer": float('inf')}
        
        # Evaluate
        results = baseline.evaluate(
            data=test_data,
            labels=test_labels,
            metrics=["eer"],
            sr=16000,
            in_domain=True,
            dataset_name=expr_name
        )
        
        return results
    
    def experiment_1_domain_generalization(self):
        """Experiment 1: Domain Generalization Stress Test (Scripted-to-Spontaneous)"""
        experiment_name = "expr1"
        logger.info("EXPERIMENT 1: Domain Generalization Stress Test")
        
        all_results = {}
        
        # Iterate over all baselines
        for baseline_name in self.baseline_names:
            logger.info(f"Running experiment 1 with baseline: {baseline_name}")
            
            model = self._train_model(
                train_datasets=['audiobook', 'news'],
                val_datasets=['audiobook', 'news'],
                expr_name=experiment_name,
                baseline_name=baseline_name
            )
            
            # Test sets
            test_sets = {
                "InDomain": ['audiobook', 'news'],
                "Spontaneous": ['interview', 'podcast', 'phonecall'],
                "RealWorld": ['movie', 'publicfigure', 'publicspeech']
            }
            
            results = {}
            for test_name, datasets in test_sets.items():
                results[test_name] = self._evaluate_model(
                    baseline=model,
                    test_datasets=datasets,
                    expr_name=experiment_name,
                    baseline_name=baseline_name
                )
            
            all_results[baseline_name] = results
        
        self.results[experiment_name] = all_results
        self.save_results()
        return all_results
    
    def experiment_2_emotional_uncanny_valley(self):
        """Experiment 2: Emotional Prosody Uncanny Valley Test"""
        experiment_name = "expr2"
        logger.info("EXPERIMENT 2: Emotional Prosody Uncanny Valley Test")
        
        all_results = {}
        
        # Iterate over all baselines
        for baseline_name in self.baseline_names:
            logger.info(f"Running experiment 2 with baseline: {baseline_name}")
            
            # Train Model_N excluding Emotional data
            train_datasets = ['audiobook', 'interview', 'movie', 'news', 'phonecall', 
                             'podcast', 'publicfigure', 'publicspeech']
            
            model = self._train_model(
                train_datasets=train_datasets,
                val_datasets=train_datasets,
                expr_name=experiment_name,
                baseline_name=baseline_name
            )
            
            # Test sets
            test_sets = {
                "Neutral": ['audiobook', 'podcast'],
                "Emotional": ['emotional']
            }
            
            results = {}
            for test_name, datasets in test_sets.items():
                results[test_name] = self._evaluate_model(
                    baseline=model,
                    test_datasets=datasets,
                    expr_name=experiment_name,
                    baseline_name=baseline_name
                )
            
            all_results[baseline_name] = results
        
        self.results[experiment_name] = all_results
        self.save_results()
        return all_results
    
    def experiment_3_sensitivity_robustness(self):
        """Experiment 3: Sensitivity vs Robustness Test"""
        experiment_name = "expr3"
        logger.info("EXPERIMENT 3: Sensitivity and Robustness Test")
        
        all_results = {}
        
        # Iterate over all baselines
        for baseline_name in self.baseline_names:
            logger.info(f"Running experiment 3 with baseline: {baseline_name}")
            
            # Train Model_C on Interview + Podcast + PublicSpeech
            train_datasets = ['interview', 'podcast', 'publicspeech']
            
            model = self._train_model(
                train_datasets=train_datasets,
                val_datasets=train_datasets,
                expr_name=experiment_name,
                baseline_name=baseline_name
            )
            
            # Test sets
            test_sets = {
                "CleanFull": ['interview', 'podcast', 'publicspeech'],
                "Partial": ['partialfake'],
                "Noisy": ['noisyspeech']
            }
            
            results = {}
            for test_name, datasets in test_sets.items():
                results[test_name] = self._evaluate_model(
                    baseline=model,
                    test_datasets=datasets,
                    expr_name=experiment_name,
                    baseline_name=baseline_name
                )
            
            all_results[baseline_name] = results
        
        self.results[experiment_name] = all_results
        self.save_results()
        return all_results
    
    def experiment_4_cross_language(self):
        """Experiment 4: Cross-Language Generalization Test"""
        experiment_name = "expr4"
        logger.info("EXPERIMENT 4: Cross-Language Generalization Test")
        
        all_results = {}
        
        # Iterate over all baselines
        for baseline_name in self.baseline_names:
            logger.info(f"Running experiment 4 with baseline: {baseline_name}")
            
            # Train Model_EN (English) - include PhoneCall/en
            en_train_datasets = ['audiobook', 'emotional', 'interview', 'movie', 
                                'podcast', 'publicfigure', 'publicspeech', 'phonecall']
            model_en = self._train_model(
                train_datasets=en_train_datasets,
                val_datasets=en_train_datasets,
                expr_name=f"{experiment_name}_en",
                subset="en",
                baseline_name=baseline_name
            )
            
            # Train Model_ZH (Chinese) - include PhoneCall/zh-cn
            zh_train_datasets = ['news', 'phonecall']
            model_zh = self._train_model(
                train_datasets=zh_train_datasets,
                val_datasets=zh_train_datasets,
                expr_name=f"{experiment_name}_zh",
                subset="zh-cn",
                baseline_name=baseline_name
            )
            
            # Test sets
            en_test_datasets = ['audiobook', 'emotional', 'interview', 'movie', 
                               'podcast', 'publicfigure', 'publicspeech', 'phonecall']
            zh_test_datasets = ['news', 'phonecall']
            
            # Evaluate both models on both test sets
            results = {
                "model_en": {
                    "en": self._evaluate_model(model_en, en_test_datasets, f"{experiment_name}_en", "en", baseline_name),
                    "zh": self._evaluate_model(model_en, zh_test_datasets, f"{experiment_name}_en", "zh-cn", baseline_name)
                },
                "model_zh": {
                    "en": self._evaluate_model(model_zh, en_test_datasets, f"{experiment_name}_zh", "en", baseline_name),
                    "zh": self._evaluate_model(model_zh, zh_test_datasets, f"{experiment_name}_zh", "zh-cn", baseline_name)
                }
            }
            
            all_results[baseline_name] = results
        
        self.results[experiment_name] = all_results
        self.save_results()
        return all_results
    
    def run_all_experiments(self):
        """Run all four experiments"""
        logger.info("Starting MDAD Benchmark Experiments")
        logger.info(f"Baselines: {', '.join(self.baseline_names)}")
        logger.info(f"Data directory: {self.data_dir}")
        
        # Run experiments
        self.experiment_1_domain_generalization()
        self.experiment_2_emotional_uncanny_valley()
        self.experiment_3_sensitivity_robustness()
        self.experiment_4_cross_language()
        
        logger.info("All experiments completed!")
        return self.results
    
    def save_results(self):
        """Save results to JSON file"""
        results_file = f"logs/results_{self.timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run MDAD Benchmark Experiments")
    parser.add_argument("-b", "--baseline", type=str, nargs="+", default=["aasist", "aasist-l", "rawnet2", "res-tssdnet", "inc-tssdnet", "rawgat-st"], 
                       choices=list(BASELINE_MAP.keys()),
                       help="Baseline model(s) to use")
    parser.add_argument("--data_dir", type=str, default="data/MDAD",
                       help="Path to the data directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("-e", "--experiment", type=str, choices=["1", "2", "3", "4", "all"],
                       default="all", help="Which experiment to run")
    
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Create and run experiments
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        baselines=args.baseline,
        device=args.device
    )
    
    if args.experiment == "all":
        runner.run_all_experiments()
    elif args.experiment == "1":
        runner.experiment_1_domain_generalization()
    elif args.experiment == "2":
        runner.experiment_2_emotional_uncanny_valley()
    elif args.experiment == "3":
        runner.experiment_3_sensitivity_robustness()
    elif args.experiment == "4":
        runner.experiment_4_cross_language()

if __name__ == "__main__":
    main()