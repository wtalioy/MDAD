from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from . import ALL_TESTS
from .runner import TestRunner

_TEST_MAP = {test.name: test for test in ALL_TESTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuadVox benchmark tests via modular CLI")
    parser.add_argument(
        "-b",
        "--baseline",
        type=str,
        nargs="+",
        default=["aasist", "aasist-l", "rawnet2", "res-tssdnet", "inc-tssdnet", "rawgat-st", "rapt"],
        help="Baseline model(s) to use",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/QuadVoxBench"), help="Path to the data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=list(_TEST_MAP.keys()) + ["all"],
        default="all",
        help="Which test to run",
    )
    return parser.parse_args()


def run_selected(runner: TestRunner, selected: str):
    tests_to_run = ALL_TESTS if selected == "all" else [_TEST_MAP[selected]]

    for test_class in tests_to_run:
        logger.info(f"Executing {test_class.name}: {test_class.__doc__}")
        test_class.run(runner)


def main():
    args = parse_args()

    logger.info("Selected baselines: {}", ", ".join(args.baseline))
    logger.info("Selected test(s): {}", args.test)

    runner = TestRunner(data_dir=str(args.data_dir), baselines=args.baseline, device=args.device)
    run_selected(runner, args.test)


if __name__ == "__main__":
    main()
