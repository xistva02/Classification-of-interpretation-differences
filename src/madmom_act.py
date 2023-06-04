import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import key_prediction_to_label
from madmom.features.key import CNNKeyRecognitionProcessor


class MadmomNetworks:
    def __init__(self, user, session):
        self.user = user
        self.session = session
        self.fps = 100
        self.dataset_path = 'dataset'
        self.data_path = f'../{self.user}/{self.session}/{self.dataset_path}'

    def initialize_folders(self, folders, specific=False):
        if specific is True:
            preset = [['madmom/beats', 'madmom/beat_act'], ['madmom/downbeats', 'madmom/downbeat_act'], ['madmom/key']]
            for i, folder in enumerate(folders):
                if folder is True:
                    for folder_name in preset[i]:
                        if not os.path.exists(f'{self.data_path}/{folder_name}'):
                            os.makedirs(f'{self.data_path}/{folder_name}')
        else:
            for folder in folders:
                if not os.path.exists(f'{self.data_path}/{folder}'):
                    os.makedirs(f'{self.data_path}/{folder}')

    def compute(self, audio_paths=False, beats=False, downbeats=False, key=False, save_act=True, save_output=True,
                beats_per_bar=None, first_only=False):

        self.initialize_folders(folders=[beats, downbeats, key], specific=True)
        # if audio_paths are provided, they have to be in a list of strings format and the file should be .npy
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        if beats_per_bar is None:
            beats_per_bar = [2, 3, 4]

        if audio_paths is False:
            audio_paths = glob(f'{self.data_path}/audio_arrays_44/*')

        if beats is True:
            dbn_beat_proc = DBNBeatTrackingProcessor(fps=self.fps)
            rnn_beat_proc = RNNBeatProcessor(fps=self.fps)
        if downbeats is True:
            rnn_downbeat_proc = RNNDownBeatProcessor(fps=self.fps)
            dbn_downbeat_proc = DBNDownBeatTrackingProcessor(fps=self.fps, beats_per_bar=beats_per_bar)
        if key is True:
            key_proc = CNNKeyRecognitionProcessor()

        for i, audio_path in tqdm(enumerate(audio_paths)):
            filename = Path(audio_path).stem
            audio_array = np.load(audio_path)
            if beats is True:
                beat_act_fun = rnn_beat_proc(audio_array)
                final_beats = dbn_beat_proc(beat_act_fun)
            if downbeats is True:
                downbeat_act_fun = rnn_downbeat_proc(audio_array)
                all_beats = dbn_downbeat_proc(downbeat_act_fun)
                df_beats = pd.DataFrame(all_beats, columns=['beat', 'pos'])
                final_downbeats = df_beats[(df_beats == 1.0).any(axis=1)]['beat'].values.tolist()
            if key is True:
                if first_only is True:
                    if i == 0:
                        key_act_fun = key_proc(audio_array)
                        final_key = key_prediction_to_label(key_act_fun)
                else:
                    key_act_fun = key_proc(audio_array)
                    final_key = key_prediction_to_label(key_act_fun)

            if save_act:
                if beats is True:
                    np.save(f'{self.data_path}/madmom/beat_act/{filename}.npy', beat_act_fun)
                if downbeats is True:
                    np.save(f'{self.data_path}/madmom/downbeat_act/{filename}.npy', downbeat_act_fun[:, 1])

            if save_output:
                if beats is True:
                    np.savetxt(f'{self.data_path}/madmom/beats/{filename}.txt', final_beats, fmt='%.2f')
                if downbeats is True:
                    np.savetxt(f'{self.data_path}/madmom/downbeats/{filename}.txt', final_downbeats, fmt='%.2f')
                if key is True:
                    if first_only is True:
                        if i == 0:
                            with open(f'{self.data_path}/madmom/key/{filename}.txt', 'w') as text_file:
                                text_file.write(final_key)
                    else:
                        with open(f'{self.data_path}/madmom/key/{filename}.txt', 'w') as text_file:
                            text_file.write(final_key)

if __name__ == '__main__':
    from config import *
    print(f'---processing {os.path.basename(__file__)}---')

    for session in sessions:
        act = MadmomNetworks(user=user, session=session)
        act.compute(beats=True, downbeats=True, key=True, save_act=True, save_output=True, first_only=True)

    print('\ndone')