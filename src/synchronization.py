import os
from os.path import basename
import sys
from glob import glob
import numpy as np
from scipy.interpolate import interp1d
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.chroma import quantized_chroma_to_CENS
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors, make_path_strictly_monotonic
from tensorflow.python.keras.models import load_model
from pathlib import Path
from madmom.audio import Signal
from tqdm import tqdm
from .utils import PreProcessor


class HiddenPrints:
    """
    Class to hide all prints to console.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Synchronization:
    def __init__(self,
                 user: str = None,
                 session: str = None,
                 movement: str = None,
                 monotonic: bool = True,
                 model_path: str = None,
                 filenames: list = None):
        """
        Class to synchronize chroma features of recordings and transfer annotation (measure positions) from a reference
        audio file to all recordings.

        Args:
            user (str)           -- name of the user/composer
            session (str)        -- name of the user/composer
            movement (str)       -- movement number
            monotonic (bool)     -- if output warping path should be monotonic or not
            model_path (str)     -- path to the beat tracking model (50 fps, 22050 Hz)
            filenames (list)     -- list of the filenames
        """

        self.feature_rate = 50
        self.step_weights = np.array([1.5, 1.5, 2.0])
        self.threshold_rec = 10 ** 6
        self.monotonic = monotonic
        self.sr = 22050
        self.fps = 50
        self.user = user
        self.session = session
        self.movement = str(movement)
        self.act_fun_type = 'beat'
        self.data_path = f'data/{self.user}/{self.session}/mov{self.movement}'

        if filenames is None:
            self.filenames = []
        else:
            self.filenames = filenames
        if model_path is None:
            self.model_path = f'model/simple_tcn_dp_skip_dilations_22_fps50.h5'
        else:
            self.model_path = model_path

    def initialize_folders(self,
                           folders: list):
        """
        Create all folders.

        Args:
            folders: list of path to folders
        """

        for folder in folders:
            if not os.path.exists(f'{self.data_path}/{folder}'):
                os.makedirs(f'{self.data_path}/{folder}')

    def get_beat_act_function(self,
                              audio_data: np.ndarray,
                              chroma_length: int,
                              model):
        """
        Get the activation function from the beat tracking model.

        Args:
            audio_data:
            chroma_length:
            model:

        Returns:
            act_fun: reshaped activation function from beat tracking
        """

        audio_signal = Signal(audio_data, self.sr)
        preprocessed = PreProcessor()(audio_signal)
        if not model:
            model = load_model(self.model_path)
        act_fun = model.predict(preprocessed[np.newaxis, ..., np.newaxis]).flatten('C')

        # paddding to ensure the same length of act function as chroma vectors
        if act_fun.size < chroma_length:
            diff = chroma_length - act_fun.size
            if diff % 2 == 0:
                pad = int(diff / 2)
                act_fun = np.concatenate((np.zeros(pad), act_fun, np.zeros(pad)))
            else:
                pad = int(diff / 2)
                act_fun = np.concatenate((np.zeros(pad), act_fun, np.zeros(pad)))
                act_fun = np.append(act_fun, np.array([0]))
        return act_fun.reshape(1, -1)

    def compute_act_fun(self,
                        activation: str = 'beat',
                        save: bool = True):
        """
        Compute activation function.

        Args:
            activation:
            save:

        Returns:
            If save is True, save the activation function as npy file, else return all activation functions as a list.
        """

        self.initialize_folders([f'{activation}_act_fun'])

        chromas = glob(f'{self.data_path}/chroma/*')
        chroma_files = []
        if self.filenames:
            # get only chromas corresponding to filenames
            for file in self.filenames:
                chroma_files.append(list(filter(lambda x: file in x, chromas))[0])
        else:
            chroma_files = chromas

        act_funs = []
        model = load_model(self.model_path)
        for chroma in chroma_files:
            filename = Path(chroma).stem
            if not os.path.exists(f'{self.data_path}/{activation}_act_fun/{filename}.npy'):
                chroma = np.load(chroma)
                audio_data = np.load(f'{self.data_path}/audio_arrays_22/{filename}.npy')
                act_fun = self.get_beat_act_function(audio_data, chroma.shape[1], model)
                if save:
                    np.save(f'{self.data_path}/{activation}_act_fun/{filename}.npy', act_fun)
                else:
                    act_funs.append(act_fun)
        if not save:
            return act_funs

    def compute_wp(self,
                   ref_chroma: np.ndarray,
                   ref_filename: str,
                   filenames: list,
                   chroma_paths: list,
                   ref_cens: np.ndarray,
                   ref_annot: list,
                   ref_act_fun: np.ndarray,
                   monotonic: bool = True,
                   save: bool = True):
        """
        Compute the warping path of given data.

        Args:
            ref_chroma:
            ref_filename:
            filenames:
            chroma_paths:
            ref_cens:
            ref_annot:
            ref_act_fun:
            monotonic:
            save:

        Returns:
            If save is True, save the output transferred measures and warping path, else return it.
        """

        list_of_measures = []
        wps = []
        for target_chroma, target_name in tqdm(zip(chroma_paths, filenames)):

            if os.path.exists(f'{self.data_path}/warping_paths/{self.act_fun_type}/'
                              f'{ref_filename}_vs_{target_name}.npy'):
                continue

            target_chroma = np.load(target_chroma)
            if ref_act_fun is not False:
                target_act_fun = np.load(
                    f'{self.data_path}/{self.act_fun_type}_act_fun/{target_name}.npy')
            target_cens = quantized_chroma_to_CENS(target_chroma, 201, 50, self.feature_rate)[0]
            opt_chroma_shift = compute_optimal_chroma_shift(ref_cens, target_cens)
            target_chroma = shift_chroma_vectors(target_chroma, opt_chroma_shift)
            print(f'Ref file: {ref_filename}, target file: {target_name}')
            with HiddenPrints():
                if ref_act_fun is not False:
                    wp = sync_via_mrmsdtw(f_chroma1=ref_chroma, f_chroma2=target_chroma,
                                          f_onset1=ref_act_fun, f_onset2=target_act_fun,
                                          input_feature_rate=self.feature_rate, step_weights=self.step_weights,
                                          threshold_rec=self.threshold_rec, verbose=False)
                else:
                    wp = sync_via_mrmsdtw(f_chroma1=ref_chroma, f_chroma2=target_chroma,
                                          input_feature_rate=self.feature_rate, step_weights=self.step_weights,
                                          threshold_rec=self.threshold_rec, verbose=False)
            if monotonic:
                wp = make_path_strictly_monotonic(wp)
            sync_measures = list(
                interp1d(wp[0] / self.feature_rate, wp[1] / self.feature_rate, kind='linear')(ref_annot))
            if save:
                if ref_act_fun is not False:
                    np.savetxt(f'{self.data_path}/sync_measures/{self.act_fun_type}/'
                               f'{target_name}.txt', sync_measures, fmt='%.6f')
                    np.save(f'{self.data_path}/warping_paths/{self.act_fun_type}/'
                            f'{ref_filename}_vs_{target_name}.npy', wp)
                else:
                    np.savetxt(f'{self.data_path}/sync_measures/chroma/'
                               f'{target_name}.txt', sync_measures, fmt='%.6f')
                    np.save(f'{self.data_path}/warping_paths/chroma/'
                            f'{ref_filename}_vs_{target_name}.npy', wp)
            else:
                list_of_measures.append(sync_measures)
                wps.append(wp)
        if not save:
            return wps, list_of_measures

    def sync_ref_targets(self,
                         ref_filename: str,
                         act_fun: bool = False,
                         act_fun_type: str = 'beat',
                         keep_ref: bool = True,
                         monotonic: bool = True,
                         save: bool = True):
        """
        Synchronize the reference recording with all target recordings.

        Args:
            ref_filename:
            act_fun:
            act_fun_type:
            keep_ref:
            monotonic:
            save:

        Returns:
            If save is True, save the output transferred measures and warping path, else return it.
        """

        self.initialize_folders(['sync_measures/chroma', f'sync_measures/{act_fun_type}',
                                 'warping_paths/chroma', f'warping_paths/{act_fun_type}'])
        chromas = glob(f'{self.data_path}/chroma/*')
        ref_chroma_path = f'{self.data_path}/chroma/{ref_filename}.npy'

        if not keep_ref:
            chromas.remove(ref_chroma_path)

        chroma_paths = []
        filenames = []
        if self.filenames:
            # get only chromas corresponding to filenames
            for file in self.filenames:
                chroma_paths.append(list(filter(lambda x: file in x, chromas))[0])
                filenames.append(Path(file).stem)
        else:
            chroma_paths = chromas
            for chroma in chroma_paths:
                filenames.append(Path(chroma).stem)

        ref_annot = np.loadtxt(glob(f'annotations/txt/{self.user}/{self.session}/*_mov{self.movement}.txt')[0])
        ref_annot = list(map(float, ref_annot))

        ref_chroma = np.load(ref_chroma_path)
        ref_cens = quantized_chroma_to_CENS(ref_chroma, 201, 50, self.feature_rate)[0]

        if act_fun:
            self.act_fun_type = act_fun_type
            ref_act_fun = np.load(f'{self.data_path}/{act_fun_type}_act_fun/{ref_filename}.npy')
        else:
            ref_act_fun = False
        if save:
            self.compute_wp(ref_chroma, ref_filename, filenames, chroma_paths, ref_cens,
                            ref_annot, ref_act_fun, monotonic=monotonic, save=save)
        else:
            list_of_sync_measures, wps = self.compute_wp(ref_chroma, ref_filename, filenames, chroma_paths, ref_cens,
                                                         ref_annot, ref_act_fun, monotonic=monotonic, save=save)
            return list_of_sync_measures, wps


if __name__ == '__main__':

    from .config import *

    print(f'---processing {basename(__file__)}---')

    for no_of_comp, user in enumerate(users):
        for session, ref_file in zip(sessions[no_of_comp], references[no_of_comp]):
            for movement in movements:
                ref_filename = f'{ref_file}_{movement}'
                print(f'---processing {user}, {session}, mov{movement}---')
                print(f'---reference: {ref_filename}---')
                sync = Synchronization(user=user, session=session, movement=movement)
                sync.sync_ref_targets(ref_filename, keep_ref=True, act_fun=False, monotonic=False, save=True)
                sync.compute_act_fun(save=True)
                sync.sync_ref_targets(ref_filename, keep_ref=True, act_fun=True, monotonic=True, save=True)

    print('---done---')
