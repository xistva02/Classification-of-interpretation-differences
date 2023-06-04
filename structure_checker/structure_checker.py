from glob import glob
import time
import os.path
import os
import numpy as np
from utils import return_diff_structures, create_output_file


def check_structure(warping_paths: np.ndarray,
                    output_path: str,
                    plot_path: str,
                    segment_divider: int = 25,
                    diff_max: int = 3,
                    diff_min: int = 0.13,
                    area_diff: int = 10,
                    output: bool = True,
                    excel: bool = True,
                    csv: bool = True,
                    debug: bool = True,
                    plotting: bool = False
                    ):
    # starts the time counter
    if debug:
        tcalcstart = time.time()

    # returns list of filenames with different structures and list of different areas from DTW
    diff_structure_list, same_structure_list = return_diff_structures(warping_paths=warping_paths,
                                                                      output_path=output_path,
                                                                      plot_path=plot_path,
                                                                      segment_divider=segment_divider,
                                                                      diff_max=diff_max,
                                                                      diff_min=diff_min,
                                                                      area_diff=area_diff,
                                                                      plotting=plotting,
                                                                      debug=debug)

    output_diff_filename = f'{user}_{session}_mov{movement}_diff_structure'
    output_same_filename = f'{user}_{session}_mov{movement}_same_structure'

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if output:
        create_output_file(diff_structure_list=diff_structure_list, path_to_output=output_path,
                           output_filename=output_diff_filename,
                           excel=excel, csv=csv)
        create_output_file(diff_structure_list=same_structure_list, path_to_output=output_path,
                           output_filename=output_same_filename,
                           excel=excel, csv=csv)

    # stops the timer and writes output to console
    if debug:
        tcalcend = time.time()
        calctime = round(tcalcend - tcalcstart, 2)

    num_of_diff_structure = len(diff_structure_list)

    print("Calculation finished!")
    if debug:
        print(f"Total calculation time: {calctime}")
    print(f"Number of different structures found: {num_of_diff_structure}")
    print(f'List of recordings with different structures: {diff_structure_list}\n')


if __name__ == '__main__':

    from config import *

    chroma_style = 'chroma'  # chroma or beat

    for no_of_comp, user in enumerate(users):
        for session, reference_name in zip(sessions[no_of_comp], references[no_of_comp]):
            for movement in movements:

                ref_file = f'{reference_name}_{movement}'
                plot_path = f'{user}/{session}/mov{movement}'
                reference_CD = reference_name[:3]
                print(f'reference_CD: {reference_CD}, {user}, {session}, mov{movement}')

                warping_paths = glob(f'../data/{user}/{session}/mov{movement}/warping_paths/{chroma_style}/*')

                check_structure(warping_paths=warping_paths,
                                plot_path=plot_path,
                                output_path=f'outputs',
                                plotting=True,
                                debug=False,
                                excel=True,
                                csv=True,
                                output=True)
                print('---segment done---')
    print('---DONE---')
