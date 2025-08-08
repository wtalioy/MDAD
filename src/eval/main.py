import os
import argparse
import warnings
import torch
from loguru import logger
from baselines import BASELINE_MAP
from cmad_datasets import DATASET_MAP

def display_results(results: dict, baseline: str, dataset: str):
    if isinstance(list(results.values())[0], dict): # PartialFake and NoisySpeech specific
        for source_name, source_results in results.items():
            logger.info("Evaluation results:")
            for metric, value in source_results.items():
                logger.info(f"({baseline} on {dataset} from {source_name}) {metric}: {value:.4f}")
    else:
        logger.info("Evaluation results:")
        for metric, value in results.items():
            logger.info(f"({baseline} on {dataset}) {metric}: {value:.4f}")

def main(args):
    warnings.filterwarnings("ignore")
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    os.makedirs("logs", exist_ok=True)
    
    for baseline in args.baseline:
        baseline = BASELINE_MAP[baseline](**vars(args))
        for dataset in args.dataset:
            dataset = DATASET_MAP[dataset](**vars(args))
            logger.info(f"Evaluating {baseline.name} on {dataset.name} with metrics: {args.metrics} in {args.mode}-domain mode")
            log_id = logger.add("logs/eval.log", rotation="100 MB", retention="60 days")

            results = None
            if args.mode == "cross":
                if not args.train_only:
                    results = dataset.evaluate(baseline, args.metrics)
            elif args.mode == "in":
                if not args.eval_only:
                    logger.info("Training baseline ...")
                    logger.remove(log_id)
                    dataset.train(baseline)
                if not args.train_only:
                    logger.info("Evaluating baseline ...")
                    results = dataset.evaluate(baseline, args.metrics, in_domain=True)
                    logger.add("logs/eval.log", rotation="100 MB", retention="60 days")
                else:
                    continue

            logger.info(f"Evaluation of {baseline.name} on {dataset.name} completed")
            if results is not None:
                display_results(results, baseline.name, dataset.name)

    logger.info(f"Evaluation completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline on dataset")
    parser.add_argument("--baseline", type=str, nargs="+", default=["aasist", "aasist-l", "rawnet2", "res-tssdnet", "inc-tssdnet"], help="Name of the baseline", choices=list(BASELINE_MAP.keys()))
    parser.add_argument("--dataset", type=str, nargs="+", default=["podcast", "phonecall", "publicspeech", "movie", "interview"], help="Name of the dataset", choices=list(DATASET_MAP.keys()))
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset")
    parser.add_argument("--mode", type=str, default="cross", help="Mode of the evaluation", choices=["cross", "in"])
    parser.add_argument("--train_only", action="store_true", help="Train the baseline only")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate the baseline only")
    parser.add_argument("--metrics", type=str, nargs="+", default=["eer"], help="Metrics to evaluate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    args = parser.parse_args()

    main(args)