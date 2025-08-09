#!/usr/bin/env python3
"""
Script to resample all audio files in the directory to a certain sample rate using librosa.
This script will recursively find all audio files and resample them in-place.
"""

import os
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from tqdm import tqdm
from loguru import logger

def is_audio_file(file_path):
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions

def resample_audio_file(file_path, target_sr=16000, backup=True):
    try:
        file_path = Path(file_path)
        
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            if not backup_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
        
        # Load the audio file
        logger.debug(f"Loading audio file: {file_path}")
        y, sr = librosa.load(file_path, sr=None)
        
        # Check if resampling is needed
        if sr == target_sr:
            logger.debug(f"File {file_path} is already at {target_sr}Hz, skipping")
            return True
        
        # Resample the audio
        logger.debug(f"Resampling from {sr}Hz to {target_sr}Hz")
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Save the resampled audio
        sf.write(file_path, y_resampled, target_sr)
        logger.debug(f"Successfully resampled {file_path} from {sr}Hz to {target_sr}Hz")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def find_audio_files(directory):
    audio_files = []
    directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"Directory {directory} does not exist")
        return audio_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if is_audio_file(file_path):
                audio_files.append(file_path)
    
    return audio_files

def main():
    parser = argparse.ArgumentParser(description='Resample all audio files in a directory to a certain sample rate')
    parser.add_argument('-d', '--data_dir', default='data/Audiobook', 
                       help='Path to the directory containing audio files (default: data/Audiobook)')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                       help='Target sample rate in Hz (default: 16000)')
    parser.add_argument('--no_backup', action='store_true',
                       help='Do not create backup files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually doing it')
    
    args = parser.parse_args()
    
    # Find all audio files
    logger.info(f"Searching for audio files in {args.data_dir}")
    audio_files = find_audio_files(args.data_dir)
    
    if not audio_files:
        logger.warning(f"No audio files found in {args.data_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    if args.dry_run:
        logger.info("DRY RUN - Would process the following files:")
        for file_path in audio_files:
            logger.info(f"  {file_path}")
        return
    
    # Process each audio file
    successful = 0
    failed = 0
    
    for file_path in tqdm(audio_files, desc="Resampling audio files"):
        if resample_audio_file(file_path, args.sample_rate, not args.no_backup):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main() 