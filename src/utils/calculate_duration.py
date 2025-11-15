import os
import json
import wave
import librosa
import csv
from pathlib import Path

def get_audio_duration(file_path):
    """Get duration of an audio file in seconds."""
    try:
        # Try using librosa first (handles more formats)
        duration = librosa.get_duration(y=file_path)
        return duration
    except Exception as e:
        try:
            # Fallback to wave module for WAV files
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e2:
            print(f"Warning: Could not get duration for {file_path}: {e2}")
            return 0.0

def load_metadata(domain_path):
    """Load metadata from meta.json file."""
    meta_path = domain_path / 'meta.json'
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata from {meta_path}: {e}")
        return None

def classify_audio_files(domain_path, metadata):
    """Classify audio files as real or fake based on metadata."""
    classification = {}
    
    if metadata is None:
        return classification
    
    for item in metadata:
        if 'audio' in item:
            audio_info = item['audio']
            if 'fake' in audio_info:
                # This is a fake audio file
                fake_info = audio_info['fake']
                if isinstance(fake_info, dict):
                    # Handle dictionary format: {"N/A": "audio/file.wav"}
                    for key, path in fake_info.items():
                        if path.startswith('audio/'):
                            filename = path.split('/')[-1]
                            classification[filename] = 'fake'
                elif isinstance(fake_info, str):
                    # Handle string format: "audio/real/id10001/file.wav"
                    if fake_info.startswith('audio/'):
                        filename = fake_info.split('/')[-1]
                        classification[filename] = 'fake'
            elif 'real' in audio_info:
                # This is a real audio file
                real_info = audio_info['real']
                if isinstance(real_info, dict):
                    # Handle dictionary format: {"N/A": "audio/file.wav"}
                    for key, path in real_info.items():
                        if path.startswith('audio/'):
                            filename = path.split('/')[-1]
                            classification[filename] = 'real'
                elif isinstance(real_info, str):
                    # Handle string format: "audio/real/id10001/file.wav"
                    if real_info.startswith('audio/'):
                        filename = real_info.split('/')[-1]
                        classification[filename] = 'real'
    
    return classification

def calculate_domain_duration(domain_path):
    """Calculate duration for a specific domain."""
    results = {
        'real': {'duration': 0.0, 'count': 0},
        'fake': {'duration': 0.0, 'count': 0}
    }
    
    # Load metadata to classify files
    metadata = load_metadata(domain_path)
    file_classification = classify_audio_files(domain_path, metadata)
    
    # Check if domain has language subdirectories (en, zh-cn)
    language_dirs = ['en', 'zh-cn']
    audio_paths = []
    
    # Check for direct audio directory
    direct_audio_path = domain_path / 'audio'
    if direct_audio_path.exists():
        audio_paths.append((direct_audio_path, False))  # False = not language subdir
    
    # Check for language subdirectories
    for lang in language_dirs:
        lang_audio_path = domain_path / lang / 'audio'
        if lang_audio_path.exists():
            audio_paths.append((lang_audio_path, True))  # True = language subdir
    
    if not audio_paths:
        return results
    
    for audio_path, is_language_subdir in audio_paths:
        # Check if there are real/fake subdirectories
        real_path = audio_path / 'real'
        fake_path = audio_path / 'fake'
        
        # Calculate real duration from real/ subdirectory
        if real_path.exists():
            for root, dirs, files in os.walk(real_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                        file_path = Path(root) / file
                        duration = get_audio_duration(str(file_path))
                        results['real']['duration'] += duration
                        results['real']['count'] += 1
        
        # Calculate fake duration from fake/ subdirectory
        if fake_path.exists():
            for root, dirs, files in os.walk(fake_path):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                        file_path = Path(root) / file
                        duration = get_audio_duration(str(file_path))
                        results['fake']['duration'] += duration
                        results['fake']['count'] += 1
        
        # Check for files directly in audio directory (use metadata classification)
        for file in audio_path.iterdir():
            if file.is_file() and file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.aac']:
                duration = get_audio_duration(file)
                
                # Use metadata classification if available
                if file.name in file_classification:
                    classification = file_classification[file.name]
                    results[classification]['duration'] += duration
                    results[classification]['count'] += 1
                else:
                    # For domains without metadata, classify based on domain name
                    if domain_path.name in ['NoisySpeech', 'PartialFake']:
                        results['fake']['duration'] += duration
                        results['fake']['count'] += 1
                    else:
                        # Default to fake for unclassified files
                        results['fake']['duration'] += duration
                        results['fake']['count'] += 1
        
        # Check for subdirectories that might be TTS systems (like in Emotional)
        for item in audio_path.iterdir():
            if item.is_dir() and item.name not in ['real', 'fake']:
                # This might be a TTS system directory - classify as fake
                for root, dirs, files in os.walk(item):
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                            file_path = Path(root) / file
                            duration = get_audio_duration(file_path)
                            results['fake']['duration'] += duration
                            results['fake']['count'] += 1
    
    return results

def format_duration(seconds):
    """Format duration in a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def main():
    data_dir = Path('data/MTAD')
    if not data_dir.exists():
        print("Error: data directory not found!")
        return
    
    # Get all domain directories
    domains = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in ['.vscode', '.git']]
    
    print("Calculating audio durations for each domain...")
    print("=" * 80)
    
    total_real = 0.0
    total_fake = 0.0
    total_real_files = 0
    total_fake_files = 0
    
    # Prepare data for CSV
    csv_data = []
    
    for domain in sorted(domains):
        print(f"\nDomain: {domain.name}")
        print("-" * 40)
        
        results = calculate_domain_duration(domain)
        
        real_avg = results['real']['duration'] / results['real']['count'] if results['real']['count'] > 0 else 0
        fake_avg = results['fake']['duration'] / results['fake']['count'] if results['fake']['count'] > 0 else 0
        
        print(f"Real audio: {format_duration(results['real']['duration'])} ({results['real']['duration']:.2f}s) - {results['real']['count']} files (avg: {format_duration(real_avg)})")
        print(f"Fake audio: {format_duration(results['fake']['duration'])} ({results['fake']['duration']:.2f}s) - {results['fake']['count']} files (avg: {format_duration(fake_avg)})")
        
        total_real += results['real']['duration']
        total_fake += results['fake']['duration']
        total_real_files += results['real']['count']
        total_fake_files += results['fake']['count']
        
        # Calculate averages
        real_avg = results['real']['duration'] / results['real']['count'] if results['real']['count'] > 0 else 0
        fake_avg = results['fake']['duration'] / results['fake']['count'] if results['fake']['count'] > 0 else 0
        
        # Add to CSV data
        csv_data.append({
            'Domain': domain.name,
            'Real_Duration_Formatted': format_duration(results['real']['duration']),
            'Real_File_Count': results['real']['count'],
            'Real_Avg_Duration_Formatted': format_duration(real_avg),
            'Fake_Duration_Formatted': format_duration(results['fake']['duration']),
            'Fake_File_Count': results['fake']['count'],
            'Fake_Avg_Duration_Formatted': format_duration(fake_avg),
            'Total_Duration_Formatted': format_duration(results['real']['duration'] + results['fake']['duration'])
        })
    
    # Calculate total averages
    total_real_avg = total_real / total_real_files if total_real_files > 0 else 0
    total_fake_avg = total_fake / total_fake_files if total_fake_files > 0 else 0
    
    # Add total row
    csv_data.append({
        'Domain': 'TOTAL',
        'Real_Duration_Formatted': format_duration(total_real),
        'Real_File_Count': total_real_files,
        'Real_Avg_Duration_Formatted': format_duration(total_real_avg),
        'Fake_Duration_Formatted': format_duration(total_fake),
        'Fake_File_Count': total_fake_files,
        'Fake_Avg_Duration_Formatted': format_duration(total_fake_avg),
        'Total_Duration_Formatted': format_duration(total_real + total_fake)
    })
    
    print("\n" + "=" * 80)
    print("TOTAL SUMMARY")
    print("=" * 80)
    print(f"Total Real Audio: {format_duration(total_real)} ({total_real:.2f}s)")
    print(f"Total Fake Audio: {format_duration(total_fake)} ({total_fake:.2f}s)")
    print(f"Total Audio: {format_duration(total_real + total_fake)} ({(total_real + total_fake):.2f}s)")
    print(f"Total Real Files: {total_real_files}")
    print(f"Total Fake Files: {total_fake_files}")
    print(f"Average Real Duration: {format_duration(total_real_avg)}")
    print(f"Average Fake Duration: {format_duration(total_fake_avg)}")
    
    # Write to CSV
    csv_filename = 'duration_summary.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Domain', 'Real_Duration_Formatted', 'Real_File_Count', 'Real_Avg_Duration_Formatted',
                     'Fake_Duration_Formatted', 'Fake_File_Count', 'Fake_Avg_Duration_Formatted',
                     'Total_Duration_Formatted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_filename}")

if __name__ == "__main__":
    main() 