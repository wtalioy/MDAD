from datasets import RAWDATASET_MAP, BaseRawDataset
from models import MODEL_MAP, BaseTTS
import argparse

def main(args):
    raw_dataset = RAWDATASET_MAP.get(args.dataset, BaseRawDataset)(data_dir=args.data_dir)
    tts_model = MODEL_MAP.get(args.model, BaseTTS)()
    raw_dataset.generate(tts_model=tts_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio data using TTS model")
    parser.add_argument("--dataset", type=str, default="cmlr", help="Name of the dataset", choices=list(RAWDATASET_MAP.keys()))
    parser.add_argument("--model", type=str, default="xttsv2", help="Name of the TTS model", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--data_dir", type=str, default="data/CMLR", help="Directory for dataset")
    
    args = parser.parse_args()
    main(args)