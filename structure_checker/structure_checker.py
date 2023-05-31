# -*- coding: utf-8 -*-
from glob import glob
import time
import os.path
import os
from utils import return_diff_structures, create_output_file


def check_structure(warping_paths, output_path, plot_path,
                    segmentdivider=25, diff_max=3, diff_min=0.13, area_diff=10,
                    output=True, debug=True, plotting=False):
    # starts the time counter
    if debug:
        tcalcstart = time.time()

    # returns list of filenames with different structures and list of different areas from DTW
    diff_structure_list, diff_regions_list = return_diff_structures(warping_paths=warping_paths,
                                                                    output_path=output_path,
                                                                    plot_path=plot_path,
                                                                    segmentdivider=segmentdivider,
                                                                    diff_max=diff_max, diff_min=diff_min,
                                                                    area_diff=area_diff,
                                                                    plotting=plotting, debug=debug)
    output_filename = f"DiffStructures_{user}_{session}_mov{movement}"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if output:
        output_dir = os.path.join(output_path, output_filename)
        create_output_file(diff_structure_list=diff_structure_list, path_to_output=output_dir)

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
                print(f'reference_CD: {reference_CD}')

                warping_paths = glob(f'../data/{user}/{session}/mov{movement}/warping_paths/{chroma_style}/*')

                check_structure(warping_paths=warping_paths,
                                plot_path=plot_path,
                                output_path=f'outputs',
                                plotting=True,
                                debug=True,
                                output=True)
                print('segment done')
    print('done')