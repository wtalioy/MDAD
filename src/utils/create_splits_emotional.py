import json
import argparse
from pathlib import Path
import random
from copy import deepcopy
import os

def create_splits(subset_dir: str, train_ratio: float):
    meta_path = os.path.join(subset_dir, "meta.json")
    with open(meta_path, 'r') as f:
        data = json.load(f)
    
    test_split = []
    rest_split = []

    for item in data:
        test_fake_audio = {method: audio for method, audio in item['audio']['fake'].items() if method == "F5TTS"}
        rest_fake_audio = {method: audio for method, audio in item['audio']['fake'].items() if method != "F5TTS"}

        if len(test_fake_audio) > 0:
            test_item = deepcopy(item)
            test_item['audio']['fake'] = test_fake_audio
            test_split.append(test_item)
        
        if len(rest_fake_audio) > 0:
            rest_item = deepcopy(item)
            rest_item['audio']['fake'] = rest_fake_audio
            rest_split.append(rest_item)
    
    random.seed(42)
    random.shuffle(rest_split)

    train_split = rest_split[:int(len(rest_split) * train_ratio)]
    dev_split = rest_split[int(len(rest_split) * train_ratio):]

    total_entries = len(train_split) + len(dev_split) + len(test_split)

    print(f"Test split size: {len(test_split)} ({len(test_split) / total_entries:.2%})")
    print(f"Train split size: {len(train_split)} ({len(train_split) / total_entries:.2%})")
    print(f"Dev split size: {len(dev_split)} ({len(dev_split) / total_entries:.2%})")

    output_dir = Path(subset_dir)
    
    with open(output_dir / "meta_train.json", 'w') as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "meta_dev.json", 'w') as f:
        json.dump(dev_split, f, ensure_ascii=False, indent=2)

    with open(output_dir / "meta_test.json", 'w') as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--subset_dir", type=str, default="data/QuadVoxBench/Emotional", help="Path to the subset directory")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.85, help="Ratio of train split")
    args = parser.parse_args()
    create_splits(args.subset_dir, args.train_ratio)