#!/usr/bin/env python3
"""
Script to reduce dataset to a certain ratio of original size.
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger

def load_meta_json(meta_path: str) -> List[Dict[str, Any]]:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def select_random_subset(entries: List[Dict[str, Any]], keep_percentage: float = 0.25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    num_to_keep = int(len(entries) * keep_percentage)
    entries_to_keep = random.sample(entries, num_to_keep)
    entries_to_remove = [e for e in entries if e not in entries_to_keep]
    
    return entries_to_keep, entries_to_remove

def get_audio_files_from_entry(entry: Dict[str, Any], base_dir: str) -> List[str]:
    audio_files = []
    
    # Add real audio file
    if "audio" in entry and "real" in entry["audio"]:
        real_path = os.path.join(base_dir, entry["audio"]["real"])
        audio_files.append(real_path)
    
    # Add fake audio files
    if "audio" in entry and "fake" in entry["audio"]:
        fake_audio = entry["audio"]["fake"]
        if isinstance(fake_audio, dict):
            for fake_path in fake_audio.values():
                full_fake_path = os.path.join(base_dir, fake_path)
                audio_files.append(full_fake_path)
        elif isinstance(fake_audio, str) and fake_audio != "":
            full_fake_path = os.path.join(base_dir, fake_audio)
            audio_files.append(full_fake_path)
    
    return audio_files

def backup_audio_files(audio_files: List[str], backup_dir: str, base_dir: str):
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in audio_files:
        if os.path.exists(file_path):
            src_path = Path(file_path)
            # Create the same directory structure in backup
            # Handle both absolute and relative paths
            relative_path = src_path.relative_to(Path(base_dir))
            dst_path = backup_path / relative_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            logger.info(f"Moved file: {file_path} -> {dst_path}")

def count_audio_files_in_entries(entries: List[Dict[str, Any]], base_dir: str) -> int:
    total_count = 0
    for entry in entries:
        audio_files = get_audio_files_from_entry(entry, base_dir)
        total_count += len(audio_files)
    return total_count

def main():
    parser = argparse.ArgumentParser(description="Reduce dataset to a certain ratio of original size")
    parser.add_argument("-d", "--data_dir", default="data/PublicSpeech",
                       help="Base directory for the dataset")
    parser.add_argument("-r", "--ratio", type=float, default=0.25,
                       help="Percentage of entries to keep (0.0 to 1.0)")
    parser.add_argument("-b", "--backup_dir", default=None,
                       help="Directory to backup removed files (if not specified, files will be deleted)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually doing it")
    parser.add_argument("-s", "--seed", type=int,
                       help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Load metadata
    meta_path = os.path.join(args.data_dir, "meta.json")
    logger.info(f"Loading metadata from: {meta_path}")
    all_entries = load_meta_json(meta_path)
    logger.info(f"Found {len(all_entries)} metadata entries")
    
    if len(all_entries) == 0:
        logger.error("No metadata entries found!")
        return
    
    # Count total audio files
    total_audio_files = count_audio_files_in_entries(all_entries, args.data_dir)
    logger.info(f"Total audio files across all entries: {total_audio_files}")
    
    # Select random subset
    entries_to_keep, entries_to_remove = select_random_subset(all_entries, args.ratio)
    
    # Count audio files in entries to keep and remove
    audio_files_to_keep = count_audio_files_in_entries(entries_to_keep, args.data_dir)
    audio_files_to_remove = count_audio_files_in_entries(entries_to_remove, args.data_dir)
    
    logger.info(f"\nSummary:")
    logger.info(f"Total metadata entries: {len(all_entries)}")
    logger.info(f"Entries to keep: {len(entries_to_keep)} ({len(entries_to_keep)/len(all_entries)*100:.1f}%)")
    logger.info(f"Entries to remove: {len(entries_to_remove)} ({len(entries_to_remove)/len(all_entries)*100:.1f}%)")
    logger.info(f"Audio files to keep: {audio_files_to_keep} ({audio_files_to_keep/total_audio_files*100:.1f}%)")
    logger.info(f"Audio files to remove: {audio_files_to_remove} ({audio_files_to_remove/total_audio_files*100:.1f}%)")
    
    # Show some examples of entries to keep
    logger.info(f"\nSample entries to keep:")
    for i, entry in enumerate(entries_to_keep[:3]):
        real_file = entry.get("audio", {}).get("real", "N/A")
        fake_count = len(entry.get("audio", {}).get("fake", {})) if isinstance(entry.get("audio", {}).get("fake"), dict) else 0
        logger.info(f"  {i+1}. Real: {os.path.basename(real_file)}, Fake files: {fake_count}")
    if len(entries_to_keep) > 3:
        logger.info(f"  ... and {len(entries_to_keep) - 3} more")
    
    # Show some examples of entries to remove
    logger.info(f"\nSample entries to remove:")
    for i, entry in enumerate(entries_to_remove[:3]):
        real_file = entry.get("audio", {}).get("real", "N/A")
        fake_count = len(entry.get("audio", {}).get("fake", {})) if isinstance(entry.get("audio", {}).get("fake"), dict) else 0
        logger.info(f"  {i+1}. Real: {os.path.basename(real_file)}, Fake files: {fake_count}")
    if len(entries_to_remove) > 3:
        logger.info(f"  ... and {len(entries_to_remove) - 3} more")
    
    if args.dry_run:
        logger.info(f"\nDRY RUN - No files were actually modified")
        return
    
    # Ask for confirmation
    response = input(f"\nProceed with removing {len(entries_to_remove)} metadata entries ({audio_files_to_remove} audio files)? (y/N): ")
    if response.lower() != 'y':
        logger.info("Operation cancelled.")
        return
    
    # Process files
    logger.info(f"\nProcessing audio files...")
    
    # Collect all audio files to remove
    all_files_to_remove = []
    for entry in entries_to_remove:
        audio_files = get_audio_files_from_entry(entry, args.data_dir)
        all_files_to_remove.extend(audio_files)
    
    backup_dir = args.backup_dir or args.data_dir + "-backup"
    logger.info(f"Backing up files to: {backup_dir}")
    backup_audio_files(all_files_to_remove, backup_dir, args.data_dir)
    
    # Update meta.json with only the entries to keep
    logger.info(f"Updating meta.json...")
    backup_meta_path = meta_path + ".backup"
    shutil.copy2(meta_path, backup_meta_path)
    logger.info(f"Backed up original meta.json to: {backup_meta_path}")
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(entries_to_keep, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nOperation completed!")
    logger.info(f"Kept: {len(entries_to_keep)} entries ({audio_files_to_keep} audio files)")
    logger.info(f"Removed: {len(entries_to_remove)} entries ({audio_files_to_remove} audio files)")
    logger.info(f"Updated meta.json with remaining entries")

if __name__ == "__main__":
    main() 