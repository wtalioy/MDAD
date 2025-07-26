import os
import argparse
from loguru import logger
from baselines import BASELINE_MAP
from cmad_datasets import DATASET_MAP

def main(args):
    os.makedirs("logs", exist_ok=True)
    log_id = logger.add("logs/eval.log", rotation="100 MB", retention="60 days")
    logger.info(f"Evaluating {args.baseline} on {args.dataset} with metrics: {args.metrics} in {args.mode} mode")
    
    baseline = BASELINE_MAP[args.baseline](**vars(args))
    dataset = DATASET_MAP[args.dataset](**vars(args))

    if args.mode == "cross-domain":
        if not args.train_only:
            results = dataset.evaluate(baseline, args.metrics)
    elif args.mode == "in-domain":
        if not args.eval_only:
            logger.info("Training baseline")
            logger.remove(log_id)
            dataset.train(baseline)
            return
        if not args.train_only:
            results = dataset.evaluate(baseline, args.metrics)
            logger.add("logs/eval.log", rotation="100 MB", retention="60 days")

    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline on dataset")
    parser.add_argument("--baseline", type=str, default="aasist", help="Name of the baseline", choices=list(BASELINE_MAP.keys()))
    parser.add_argument("--dataset", type=str, default="publicfigure", help="Name of the dataset", choices=list(DATASET_MAP.keys()))
    parser.add_argument("--subset", type=str, default="en", help="Subset of the dataset", choices=["en", "zh-cn"])
    parser.add_argument("--mode", type=str, default="cross-domain", help="Mode of the evaluation", choices=["cross-domain", "in-domain"])
    parser.add_argument("--train_only", action="store_true", help="Train the baseline only")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate the baseline only")
    parser.add_argument("--metrics", type=str, nargs="+", default=["eer"], help="Metrics to evaluate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    args = parser.parse_args()

    main(args)