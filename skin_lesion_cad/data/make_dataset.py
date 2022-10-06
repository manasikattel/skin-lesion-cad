"""Script to download original project's data from moodle URLs
and save it in the data/ folder"""
import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

CHALLENGE_1_TRAIN = 'http://dixie.udg.edu/CAD2022/Chall1/train.tgz'
CHALLENGE_1_VAL = 'http://dixie.udg.edu/CAD2022/Chall1/val.tgz'
CHALLENGE_2_TRAIN = 'http://dixie.udg.edu/CAD2022/Chall2/train.tgz'
CHALLENGE_2_VAL = 'http://dixie.udg.edu/CAD2022/Chall2/val.tgz'

RAW_DATA_PATH = 'data/raw'
data_urls = {'chall1': {'train': CHALLENGE_1_TRAIN, 'val': CHALLENGE_1_VAL},
             'chall2': {'train': CHALLENGE_2_TRAIN, 'val': CHALLENGE_2_VAL}}


def download_file(url: str, filename: str):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


if __name__ == '__main__':
    # Download archives into corresponding folders
    for chname in data_urls.keys():
        for split in data_urls[chname].keys():
            url = data_urls[chname][split]
            filename = f'{RAW_DATA_PATH}/{chname}/' + '_'.join([url.split('/')[-2],
                                                                url.split('/')[-1]])
            # make sure all parent folders exist
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

            print('Downloading ', filename)
            download_file(url, filename)

            # extract files
            print('Extracting ', filename)
            file = tarfile.open(filename)
            file.extractall(f'{RAW_DATA_PATH}/{chname}')
            file.close()

            os.remove(filename)
