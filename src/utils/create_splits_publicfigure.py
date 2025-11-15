import json
import argparse
from pathlib import Path
import random
import os

def create_splits(subset_dir: str, train_ratio: float):
    meta_path = os.path.join(subset_dir, "meta.json")
    with open(meta_path, 'r') as f:
        data = json.load(f)
    
    speaker_ids = {}
    for item in data:
        speaker_id = item.get('speaker_id', 'unknown')
        if speaker_id not in speaker_ids:
            speaker_ids[speaker_id] = 0
        speaker_ids[speaker_id] += 1

    print(f"Total entries: {len(data)}")
    
    print(f"Total speakers: {len(speaker_ids)}")
    print("Speaker ID counts:")
    for speaker_id, count in sorted(speaker_ids.items()):
        print(f"  {speaker_id}: {count} entries")

    test_speaker_ids = ["Alec Guinness", "Bernie Sanders", "Christopher Hitchens", "Bill Clinton", "Boris Johnson", "Lyndon Johnson"]
    test_split = [item for item in data if item['speaker_id'] in test_speaker_ids]
    rest_split = [item for item in data if item['speaker_id'] not in test_speaker_ids]
    
    print(f"Test split size: {len(test_split)} ({len(test_split) / len(data):.2%})")
    
    random.seed(42)
    random.shuffle(rest_split)

    train_split = rest_split[:int(len(rest_split) * train_ratio)]
    dev_split = rest_split[int(len(rest_split) * train_ratio):]

    print(f"Train split size: {len(train_split)} ({len(train_split) / len(data):.2%})")
    print(f"Dev split size: {len(dev_split)} ({len(dev_split) / len(data):.2%})")

    output_dir = Path(subset_dir)
    
    with open(output_dir / "meta_train.json", 'w') as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "meta_dev.json", 'w') as f:
        json.dump(dev_split, f, ensure_ascii=False, indent=2)

    with open(output_dir / "meta_test.json", 'w') as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--subset_dir", type=str, default="data/QuadVoxBench/PublicFigure", help="Path to the data directory")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.85, help="Ratio of train split")
    args = parser.parse_args()
    create_splits(args.subset_dir, args.train_ratio)