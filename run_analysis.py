from os.path import basename
from glob import glob
import pandas as pd
from src.get_audio_arrays import get_audio_npy
from src.chroma import ChromaVectors
from src.synchronization import Synchronization
from structure_checker.structure_checker import check_structure
from classification import get_logger, run_classification
from argparse import ArgumentParser
from src.config import *

if __name__ == '__main__':

    def parse_args():
        """
        Parser with flags.
        """
        parser = ArgumentParser()
        parser.add_argument('-d', '--debug', type=bool, required=False, default=True,
                            help='Debug mode')
        return parser.parse_args()


    print(f'...processing file: {basename(__file__)}...')

    args = parse_args()
    debug = args.debug
    metrics = {}

    for no_of_comp, user in enumerate(users):
        for session, ref_file in zip(sessions[no_of_comp], references[no_of_comp]):
            for movement in movements:

                print(f'\n--------processing {user}, {session}--------')

                ################################################
                print(f'---audio files---')
                audio_path = f'F:/_KInG/_king_database/{user}/{session}'  # path to the audio files
                get_audio_npy(audio_path=audio_path, user=user, session=session, movement=movement, srs=[22050])
                print(f'\n')

                ################################################
                print(f'---chroma files---')
                chrV = ChromaVectors(user=user, session=session, movement=movement, verbose=False)
                chrV.run(save=True)
                print(f'\n')

                ################################################
                print(f'---synchronization---')
                ref_filename = f'{ref_file}_{movement}'
                print(f'---processing {user}, {session}, mov{movement}---')
                print(f'---reference: {ref_filename}---')
                sync = Synchronization(user=user, session=session, movement=movement)
                sync.sync_ref_targets(ref_filename, keep_ref=True, act_fun=False, monotonic=False, save=True)
                # sync.compute_act_fun(save=True)
                # sync.sync_ref_targets(ref_filename, keep_ref=True, act_fun=True, monotonic=True, save=True)
                print(f'\n')

                ################################################
                print(f'---structure checker---')
                plot_path = f'{user}/{session}/mov{movement}'
                reference_CD = ref_filename[:3]
                print(f'reference_CD: {reference_CD}, {user}, {session}, mov{movement}')
                warping_paths = glob(f'data/{user}/{session}/mov{movement}/warping_paths/chroma/*')
                check_structure(user=user,
                                session=session,
                                movement=movement,
                                warping_paths=warping_paths,
                                plot_path=plot_path,
                                output_path=f'structure_checker/outputs',
                                plotting=True,
                                debug=False,
                                excel=True,
                                csv=True,
                                output=True,
                                rewrite=False)
                print(f'\n')

    if debug:
        logger = get_logger()
    for label in labels:
        metrics[f'{label}'] = dict()
        for scenario in scenarios:
            metrics[f'{label}'][f'{scenario}'] = dict()
            metrics[f'{label}'][f'{scenario}']['fscore'] = dict()
            metrics[f'{label}'][f'{scenario}']['std'] = dict()
            for no_of_comp, user in enumerate(users):
                for session in sessions[no_of_comp]:
                    for movement in movements:

                        if user == 'Dvorak' and session == 'No.14' and scenario == 'movements':
                            continue
                        if user == 'Smetana' and session == 'No.2' and scenario == 'movements':
                            continue
                        if user == 'Dvorak' and session == 'No.14' and movement == '2' and scenario == 'motifs':
                            continue
                        if user == 'Smetana' and session == 'No.2' and scenario == 'motifs':
                            continue
                        if user == 'Dvorak' and session == 'No.14' and movement == '2' and scenario == 'measures':
                            continue
                        if user == 'Smetana' and session == 'No.2' and movement == '4' and scenario == 'measures':
                            continue

                        if debug:
                            fscore, std = run_classification(user=user,
                                                             session=session,
                                                             movement=movement,
                                                             scenario=scenario,
                                                             label=label, stats=True, save=True, debug=debug)
                        print(f'Fscore: {fscore}, std: {std}')

                        metrics[f'{label}'][f'{scenario}']['fscore'][f'{user}_{session}_mov{movement}'] = round(fscore, 3)
                        metrics[f'{label}'][f'{scenario}']['std'][f'{user}_{session}_mov{movement}'] = round(std, 3)

    for label in labels:
        for scenario in scenarios:
            df_fscore = pd.DataFrame.from_dict({f'fscore': metrics[f'{label}'][f'{scenario}']['fscore']})
            df_std = pd.DataFrame.from_dict({f'std': metrics[f'{label}'][f'{scenario}']['std']})

            df_merged = pd.DataFrame(pd.concat([df_fscore, df_std], axis=1))
            df_merged.to_csv(f'results/fscore_{scenario}_{label}.csv', index=True)
            df_merged.to_excel(f'results/fscore_{scenario}_{label}.xlsx', index=True)

    print('\n---done---')
