import requests
import os
from tqdm import tqdm

def download_from_url(url, save_dir, filename=None):
    if filename is None:
        filename = url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        return filepath
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192
        with open(filepath, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=filename
        ) as bar:
            for data in r.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
    return filepath