#!/usr/bin/env python3
"""
Script to clean up backup files created during audio resampling.
This script safely removes .backup files from the Audiobook directory.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger

def find_backup_files(directory):
    """
    Recursively find all backup files in the given directory.
    
    Args:
        directory (str): Root directory to search
    
    Returns:
        list: List of backup file paths
    """
    backup_files = []
    directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"Directory {directory} does not exist")
        return backup_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.backup'):
                file_path = Path(root) / file
                backup_files.append(file_path)
    
    return backup_files

def verify_original_file_exists(backup_path):
    """
    Check if the original file (without .backup extension) exists.
    
    Args:
        backup_path (Path): Path to the backup file
    
    Returns:
        bool: True if original file exists, False otherwise
    """
    original_path = backup_path.with_suffix('')  # Remove .backup extension
    return original_path.exists()

def clean_backup_files(directory, dry_run=False, verify_originals=True, force=False):
    """
    Clean up backup files in the specified directory.
    
    Args:
        directory (str): Directory to search for backup files
        dry_run (bool): If True, only show what would be deleted
        verify_originals (bool): If True, only delete backups where original files exist
        force (bool): If True, delete all backup files without verification
    
    Returns:
        tuple: (deleted_count, skipped_count, error_count)
    """
    logger.info(f"Searching for backup files in {directory}")
    backup_files = find_backup_files(directory)
    
    if not backup_files:
        logger.info("No backup files found")
        return 0, 0, 0
    
    logger.info(f"Found {len(backup_files)} backup files")
    
    deleted_count = 0
    skipped_count = 0
    error_count = 0
    
    for backup_path in tqdm(backup_files, desc="Processing backup files"):
        try:
            # Check if original file exists (unless force is True)
            if verify_originals and not force:
                if not verify_original_file_exists(backup_path):
                    logger.warning(f"Skipping {backup_path} - original file not found")
                    skipped_count += 1
                    continue
            
            if dry_run:
                logger.info(f"Would delete: {backup_path}")
                deleted_count += 1
            else:
                # Delete the backup file
                backup_path.unlink()
                logger.debug(f"Deleted: {backup_path}")
                deleted_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {backup_path}: {str(e)}")
            error_count += 1
    
    return deleted_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Clean up backup files from audio resampling')
    parser.add_argument('-d', '--data_dir', default='data/Audiobook', 
                       help='Path to the directory containing backup files (default: data/Audiobook)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--no-verify', action='store_true',
                       help='Do not verify that original files exist before deleting backups')
    parser.add_argument('--force', action='store_true',
                       help='Force deletion of all backup files without verification')
    
    args = parser.parse_args()
    
    # Safety check
    if not args.dry_run and not args.force:
        logger.warning(f"This will delete backup files from {args.data_dir}")
        logger.warning("This action cannot be undone!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Operation cancelled.")
            return
    
    # Clean backup files
    deleted, skipped, errors = clean_backup_files(
        args.data_dir, 
        dry_run=args.dry_run,
        verify_originals=not args.no_verify,
        force=args.force
    )
    
    # Print summary
    if args.dry_run:
        logger.info(f"DRY RUN SUMMARY: Would delete {deleted} files, skip {skipped} files, {errors} errors")
    else:
        logger.info(f"CLEANUP SUMMARY: Deleted {deleted} files, skipped {skipped} files, {errors} errors")
    
    if errors > 0:
        logger.warning(f"There were {errors} errors during cleanup. Check the logs above.")

if __name__ == "__main__":
    main() 