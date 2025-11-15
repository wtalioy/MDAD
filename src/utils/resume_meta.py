"""
Script to update meta.json with generated audio information from log (for unexpected interrupt in generation)
"""

import argparse
import json
import re
from loguru import logger

def parse_log_file(log_path):
    mappings = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_item = None
    current_method = None
    
    for line in lines:
        # Look for processing item with method
        process_match = re.search(r'Processing item (\d+) with assigned combination: (.+)', line)
        if process_match:
            current_item = int(process_match.group(1))
            current_method = process_match.group(2)
            continue
        
        # Look for generated audio path
        if current_item is not None and "Generated audio at" in line:
            path_match = re.search(r'Generated audio at (.+)', line)
            if path_match:
                audio_path = path_match.group(1).strip()
                # Convert to relative path from data/Interview/
                if audio_path.startswith('data/Interview/'):
                    audio_path = audio_path[len('data/Interview/'):]
                mappings[current_item] = {
                    'method': current_method,
                    'path': audio_path
                }
        
        # Look for successful generation (reset current tracking)
        if current_item is not None and f"Successfully generated audio for item {current_item}" in line:
            current_item = None
            current_method = None
    
    return mappings

def update_meta_json(meta_path, mappings):
    with open(meta_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Original meta.json has {len(data)} entries")
    logger.info(f"Found {len(mappings)} successful generations")
    
    updated_count = 0
    
    for item_num, info in mappings.items():
        # Item numbers are 1-indexed, but JSON array is 0-indexed
        json_index = item_num - 1
        
        if json_index < len(data):
            entry = data[json_index]
            
            # Add fake audio information to the entry
            if 'audio' not in entry:
                entry['audio'] = {}
            
            if 'fake' not in entry['audio']:
                entry['audio']['fake'] = {}
            
            # Add the generated audio path with the method as key
            entry['audio']['fake'][info['method']] = info['path']
            updated_count += 1
        else:
            logger.warning(f"Warning: Item {item_num} (index {json_index}) is out of range for meta.json")
    
    logger.info(f"Updated {updated_count} entries in meta.json")
    
    # Write the updated data back
    with open(meta_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return updated_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_path", type=str)
    parser.add_argument("-m", "--meta_path", type=str, default="data/QuadVoxBench/Interview/meta.json")
    args = parser.parse_args()
    
    logger.info("Parsing log file...")
    mappings = parse_log_file(args.log_path)
    
    logger.info(f"Found {len(mappings)} successful generations")
    logger.info("Sample mappings:")
    for item, info in sorted(mappings.items())[:5]:
        logger.info(f"  Item {item}: {info['method']} -> {info['path']}")
    
    logger.info("Updating meta.json...")
    updated_count = update_meta_json(args.meta_path, mappings)
    
    logger.info(f"Successfully updated {updated_count} entries in {args.meta_path}")

if __name__ == "__main__":
    main()