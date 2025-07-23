import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
from loguru import logger
from generation.models import TTS_MODEL_MAP, VC_MODEL_MAP
from generation.datasets import RAWDATASET_MAP


def main(args):
    raw_dataset = RAWDATASET_MAP[args.dataset](**vars(args))
    tts_models = [TTS_MODEL_MAP[model]() for model in args.tts_model]
    vc_models = [VC_MODEL_MAP[model]() for model in args.vc_model]
    raw_dataset.generate(tts_models=tts_models, vc_models=vc_models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio data")
    parser.add_argument("--dataset", type=str, default="phonecall", help="Name of the dataset", choices=list(RAWDATASET_MAP.keys()))
    parser.add_argument("--tts_model", type=str, nargs="+", default=["gpt4omini", "xttsv2", "melotts", "bark", "yourtts"], help="Name of the TTS model", choices=list(TTS_MODEL_MAP.keys()))
    parser.add_argument("--vc_model", type=str, nargs="+", default=["knnvc", "freevc", "openvoice"], help="Name of the VC model", choices=list(VC_MODEL_MAP.keys()))
    parser.add_argument("--split", type=str, default="en", help="Split of the dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory for dataset")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/generation.log", rotation="100 MB", retention="60 days")
    from datetime import datetime
    start_time = datetime.now()
    logger.info("Generation starts")
    logger.info(f"Generating fake audio data for {args.dataset} with TTS models: {args.tts_model} and VC models: {args.vc_model}")

    main(args)

    end_time = datetime.now()
    logger.info(f"Complete! Generation which started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} took {end_time - start_time}")