import os.path
from os.path import basename
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
from .utils import ffmpeg_load_audio

def process_audio_to_npy(audio_paths: list,
                         out_dir: str,
                         sr: int):
    """
    Process audio recordings to numpy npy files.

    Args:
        audio_paths (list):
        out_dir (str):
        sr (int):
    """

    for audio_path in tqdm(audio_paths):
        filename = Path(audio_path).stem
        if not os.path.exists(f'{out_dir}/{filename}.npy'):
            y, _ = ffmpeg_load_audio(audio_path, sr=sr, mono=True)
            np.save(f'{out_dir}/{filename}.npy', y)


def get_audio_npy(audio_path: str,
                  user: str,
                  session: str,
                  movement: str,
                  out_dir: str = None,
                  srs: list = None):
    """
    Process audio recordings and save them as npy files.

    Args:
        audio_path (str):
        user (str):
        session (str):
        movement (str):
        out_dir (str):
        srs (list):
    """
    if srs is None:
        srs = [22050]
    audio_paths = glob(f'{audio_path}/*/*_{movement}*')
    for sr in srs:
        if out_dir is None:
            out_dir = f'data/{user}/{session}/mov{movement}/audio_arrays_{str(sr)[0:2]}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        process_audio_to_npy(audio_paths, out_dir, sr)


if __name__ == '__main__':

    from .config import *

    print(f'...processing file: {basename(__file__)}...')

    for i, user in enumerate(users):
        for session in sessions[i]:
            print(f'\n...processing {user}, {session}...')
            for movement in movements:
                audio_path = f'F:/_KInG/_king_database/{user}/{session}'
                out_dir = f'../data/{user}/{session}/mov{movement}/audio_arrays_22'
                get_audio_npy(audio_path=audio_path, user=user, session=session, movement=movement, out_dir=out_dir,
                              srs=[22050])
    print('---done---')
