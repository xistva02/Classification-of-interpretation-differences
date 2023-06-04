import os
from get_audio_arrays import get_audio_npy
from chroma import ChromaVectors
from piano_roll import get_piano_rolls
from synchronization import Synchronization
from madmom_act import MadmomNetworks
from chords import Chords

from config import *

if __name__ == '__main__':

    print(f'processing file: {os.path.basename(__file__)}...')

    for i, user in enumerate(users):
        for session in sessions[i]:
            print(f'\n--------processing {user}, {session}--------')
            ################################################
            print(f'---audio files---')
            audio_path = f'F:/_KInG/_king_database/{user}/{session}'
            get_audio_npy(audio_path=audio_path, user=user, session=session, srs=[22050])
            ################################################
            print(f'---chroma files---')
            chrV = ChromaVectors(user=user, session=session)
            chrV.run(save=True)
            ################################################
            print(f'---synchronization---')
            ref_filename = '../'
            sync = Synchronization(user=user, session=session)
            sync.compute_act_fun(save=True)
            sync.sync_ref_targets(ref_filename, keep_ref=True, monotonic=True, save=True)
            sync.sync_ref_targets(ref_filename, keep_ref=True, act_fun=True, monotonic=True, save=True)
            ################################################
            print(f'---midis and piano rolls---')
            get_piano_rolls(user=user, session=session)
            ################################################
            print(f'---chords---')
            chords = Chords(user=user, session=session)
            chords.compute_chords(save_chroma=True, save_chords=True)
            ################################################
            print(f'---madmom features---')
            act = MadmomNetworks(user=user, session=session)
            act.compute(beats=True, downbeats=True, key=True, save_act=True, save_output=True, first_only=True)

    print('\ndone')
