import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generation.datasets import RAWDATASET_MAP
from generation.models import TTS_MODEL_MAP, VC_MODEL_MAP
import argparse

def main(args):
    raw_dataset = RAWDATASET_MAP[args.dataset](data_dir=args.data_dir)
    tts_models = [TTS_MODEL_MAP[model]() for model in args.tts_model]
    vc_models = [VC_MODEL_MAP[model]() for model in args.vc_model]
    raw_dataset.generate(tts_models=tts_models, vc_models=vc_models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio data using TTS model")
    parser.add_argument("--dataset", type=str, default="news", help="Name of the dataset", choices=list(RAWDATASET_MAP.keys()))
    parser.add_argument("--tts_model", type=str, nargs="+", default=["xttsv2", "yourtts", "melotts", "tacotron2", "bark"], help="Name of the TTS model", choices=list(TTS_MODEL_MAP.keys()))
    parser.add_argument("--vc_model", type=str, nargs="+", default=["knnvc", "freevc", "openvoice"], help="Name of the VC model", choices=list(VC_MODEL_MAP.keys()))
    parser.add_argument("--data_dir", type=str, default=None, help="Directory for dataset")
    
    args = parser.parse_args()
    main(args)