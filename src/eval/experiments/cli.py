from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from . import ALL_EXPERIMENTS
from .runner import ExperimentRunner

_EXPERIMENT_MAP = {exp.name: exp for exp in ALL_EXPERIMENTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDAD benchmark experiments via modular CLI")
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        nargs="+",
        default=["aasist", "aasist-l", "rawnet2", "res-tssdnet", "inc-tssdnet", "rawgat-st"],
        help="Baseline model(s) to use",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/MDAD"), help="Path to the data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        choices=list(_EXPERIMENT_MAP.keys()) + ["all"],
        default="all",
        help="Which experiment to run",
    )
    return parser.parse_args()


def run_selected(runner: ExperimentRunner, selected: str):
    experiments_to_run = ALL_EXPERIMENTS if selected == "all" else [_EXPERIMENT_MAP[selected]]

    for experiment_class in experiments_to_run:
        logger.info(f"Executing {experiment_class.name}: {experiment_class.__doc__}")
        experiment_class.run(runner)


def main():
    args = parse_args()

    logger.info("Selected baselines: {}", ", ".join(args.baseline))
    logger.info("Selected experiment(s): {}", args.experiment)

    runner = ExperimentRunner(data_dir=str(args.data_dir), baselines=args.baseline, device=args.device)
    run_selected(runner, args.experiment)


if __name__ == "__main__":
    main()
