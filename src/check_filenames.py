from glob import glob
import pandas as pd
from pathlib import Path
from config import *
from collections import Counter


def check_all_filenames(path_to_metadata, path_to_audio, session):
    audio_files = glob(f'{path_to_audio}/*')
    audio_filenames = []
    for audio_file in audio_files:
        audio_filenames.append(Path(audio_file).stem)

    df_metadata = pd.read_csv(path_to_metadata)
    df_metadata = df_metadata.loc[df_metadata.comp == f'{session}']
    metadata_filenames = df_metadata['filename'].values
    return sorted(audio_filenames), sorted(metadata_filenames)


if __name__ == '__main__':
    for user in users:
        for session in sessions:
            audio_filenames, df_metadata = check_all_filenames(f'../metadata/_dataset_strings_{user}.csv',
                                                               f'F:/_KInG/_king_database/{user}/{session}/',
                                                               session)
            print(f'All files are present in: user: {user}, session: {session}: ---{audio_filenames == df_metadata}---')
            if (audio_filenames == df_metadata) is False:
                print(f'Audio filenames: {audio_filenames}')
                print(f'Metadata filenames: {df_metadata}')
                result = list((Counter(audio_filenames) - Counter(df_metadata)).elements())
                wh = 'Audio filenames'
                if not result:
                    result = list((Counter(df_metadata) - Counter(audio_filenames)).elements())
                    wh = 'Metadata filenames'
                print(f'Result: {result} on top of {wh}')
