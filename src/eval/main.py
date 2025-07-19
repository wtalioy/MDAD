import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
from loguru import logger
from eval.baselines import BASELINE_MAP
from eval.datasets import DATASET_MAP

def main(args):
    baseline = BASELINE_MAP[args.baseline](wav2vec_model_path=args.wav2vec_model_path)
    dataset = DATASET_MAP[args.dataset]()
    dataset.evaluate(baseline, args.metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline on dataset")
    parser.add_argument("--baseline", type=str, default="ardetect", help="Name of the baseline", choices=list(BASELINE_MAP.keys()))
    parser.add_argument("--dataset", type=str, default="public", help="Name of the dataset", choices=list(DATASET_MAP.keys()))
    parser.add_argument("--metrics", type=str, nargs="+", default=["eer"], help="Metrics to evaluate", choices=["eer"])
    parser.add_argument("--wav2vec_model_path", type=str, default="/home/ruimingwang/AR-Detect/cache/extractors/wav2vec2-xls-r-2b", help="Path to the wav2vec model")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/eval.log", rotation="100 MB", retention="60 days")
    logger.info(f"Evaluating {args.baseline} on {args.dataset} with metrics: {args.metrics}")

    main(args)