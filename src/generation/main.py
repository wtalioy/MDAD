import os
import argparse
from loguru import logger
from .models import TTS_MODEL_MAP, VC_MODEL_MAP
from .raw_datasets import RAWDATASET_MAP
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Generate audio data")
    parser.add_argument("-d", "--dataset", type=str, nargs="+", default=["partialfake"], help="Name of the dataset", choices=list(RAWDATASET_MAP.keys()))
    parser.add_argument("-t", "--tts_model", type=str, nargs="+", default=["gpt4omini", "xttsv2", "melotts", "bark", "yourtts"], help="Name of the TTS model", choices=list(TTS_MODEL_MAP.keys()))
    parser.add_argument("-v", "--vc_model", type=str, nargs="+", default=["openvoice"], help="Name of the VC model", choices=list(VC_MODEL_MAP.keys()))
    parser.add_argument("-s", "--subset", type=str, default="zh-cn", help="Subset of the dataset")
    parser.add_argument("--data_dir", type=str, default="data/MDAD", help="Directory for dataset")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    log_id = logger.add("logs/generation.log", rotation="20 MB", retention="60 days")
    start_time = datetime.now()
    logger.info(f"Generating fake audio data for datasets: {args.dataset} with TTS models: {args.tts_model} and VC models: {args.vc_model}")
    logger.remove(log_id)

    tts_models = [TTS_MODEL_MAP[model]() for model in args.tts_model]
    vc_models = [VC_MODEL_MAP[model]() for model in args.vc_model]
    for dataset in args.dataset:
        logger.info(f"Generating fake audio data for {dataset} ...")
        raw_dataset = RAWDATASET_MAP[dataset](**vars(args))
        raw_dataset.generate(tts_models=tts_models, vc_models=vc_models)
        logger.info(f"Generation for {dataset} completed")

    end_time = datetime.now()
    logger.add("logs/generation.log", rotation="20 MB", retention="60 days")
    logger.info(f"Generation started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} completed in {end_time - start_time}")

if __name__ == "__main__":
    main()