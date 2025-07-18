import argparse
from eval.baselines import BASELINE_MAP
from eval.datasets import DATASET_MAP

def main(args):
    baseline = BASELINE_MAP[args.baseline]
    dataset = DATASET_MAP[args.dataset]()
    dataset.evaluate(baseline, args.metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline on dataset")
    parser.add_argument("--baseline", type=str, default="ardetect", help="Name of the baseline", choices=list(BASELINE_MAP.keys()))
    parser.add_argument("--dataset", type=str, default="public", help="Name of the dataset", choices=list(DATASET_MAP.keys()))
    parser.add_argument("--metrics", type=str, nargs="+", default=["eer"], help="Metrics to evaluate", choices=["eer"])
    args = parser.parse_args()
    main(args)