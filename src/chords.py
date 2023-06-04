import os
import numpy as np
from glob import glob
from pathlib import Path
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from madmom.audio.chroma import DeepChromaProcessor
from tqdm import tqdm

class Chords:
    def __init__(self, user, session):
        self.user = user
        self.session = session
        self.dataset_path = 'dataset'
        self.data_path = f'../{self.user}/{self.session}/{self.dataset_path}'

    def initialize_folders(self, folders):
        for folder in folders:
            if not os.path.exists(f'{self.data_path}/{folder}'):
                os.makedirs(f'{self.data_path}/{folder}')

    def compute_chords(self, audio_paths=False, save_chroma=True, save_chords=True):
        self.initialize_folders(['deep_chroma', 'chords'])
        # if audio_paths are provided, they have to be in a list of strings format and the file should be .npy

        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        if audio_paths is False:
            audio_paths = glob(f'{self.data_path}/audio_arrays_44/*')

        chroma_proc = DeepChromaProcessor()
        chords_proc = DeepChromaChordRecognitionProcessor()

        for audio_path in tqdm(audio_paths):
            filename = Path(audio_path).stem
            if not os.path.exists(f'{self.data_path}/deep_chroma/{filename}.npy'):
                audio_array = np.load(audio_path)
                deep_chroma = chroma_proc(audio_array)
                final_chords = chords_proc(deep_chroma)
                if save_chroma:
                    np.save(f'{self.data_path}/deep_chroma/{filename}.npy', deep_chroma)
                if save_chords:
                    np.savetxt(f'{self.data_path}/chords/{filename}.txt', final_chords, fmt="%s")


if __name__ == '__main__':
    from config import *
    print(f'---processing {os.path.basename(__file__)}---')

    for session in sessions:
        chords = Chords(user=user, session=session)
        chords.compute_chords(save_chroma=True, save_chords=True)

    print('\ndone')

