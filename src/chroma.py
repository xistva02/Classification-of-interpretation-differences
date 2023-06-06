import os
from os.path import basename
import numpy as np
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.utils import estimate_tuning
from glob import glob
from pathlib import Path
from tqdm import tqdm


class ChromaVectors:
    def __init__(self,
                 user: str,
                 session: str,
                 movement: str,
                 verbose: bool = False):
        """
        Class to compute chroma features (vectors) using synctoolbox functionalities.

        Args:
            user (str)            -- name of the user/composer
            session (str)         -- name of the user/composer
            movement (str)        -- movement number
        """

        self.user = user
        self.session = session
        self.movement = str(movement)
        self.verbose = verbose
        self.feature_rate = 50
        self.sr = 22050
        self.mono = True
        self.normalize = True

        if not os.path.exists(f'data/{self.user}/{self.session}/mov{movement}/chroma'):
            os.makedirs(f'data/{self.user}/{self.session}/mov{movement}/chroma')

    def get_chroma_features_from_audio(self,
                                       audio_path: str):
        """
        Get chroma from audio recording (path)

        Args:
            audio_path (str):

        Returns:
            f_chroma_quantized: final chroma output
        """

        audio_data = np.load(audio_path)
        tuning_offset = estimate_tuning(audio_data, self.sr)
        f_pitch = audio_to_pitch_features(f_audio=audio_data, Fs=self.sr, tuning_offset=int(tuning_offset),
                                          feature_rate=self.feature_rate, verbose=self.verbose)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        return f_chroma_quantized

    def run(self,
            save: bool = True):
        """
        Main function of the class to compute chroma vectors.

        Args:
            save (bool):

        Returns:
            If save == True, save the resulting chromas as npy files, else return all chromas as list
        """

        chromas = []
        audio_arrays = glob(f'data/{self.user}/{self.session}/mov{self.movement}/audio_arrays_22/*')
        for audio_array in tqdm(audio_arrays):
            filename = Path(audio_array).stem
            if not os.path.exists(f'data/{self.user}/{self.session}/mov{self.movement}/chroma/{filename}.npy'):
                chroma = self.get_chroma_features_from_audio(audio_array)
                if save:
                    np.save(f'data/{self.user}/{self.session}/mov{self.movement}/chroma/{filename}.npy', chroma)
                else:
                    chromas.append(chroma)
        if not save:
            return chromas


if __name__ == '__main__':

    from .config import *

    print(f'...processing file: {basename(__file__)}...')

    for no_of_comp, user in enumerate(users):
        for session in sessions[no_of_comp]:
            print(f'\n...processing {user}, {session}...')
            for movement in movements:
                chrV = ChromaVectors(user=user, session=session, movement=movement, verbose=False)
                chrV.run(save=True)

    print('---done---')
