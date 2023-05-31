import pandas as pd
from matplotlib import pyplot as plt
import os.path
import tikzplotlib
import numpy as np
import os


def return_diff_structures(warping_paths, output_path, plot_path,
                           segmentdivider=25, diff_max=3, diff_min=0.13, area_diff=10,
                           plotting=False, debug=True):
    duplicate_pairs_list = []
    diff_regions_list = []
    num_of_pairs = len(warping_paths)

    # Checking every combination if it has the same structure
    for i, warping_path in enumerate(warping_paths, start=1):

        path_split = warping_path.split('_')

        ref_split = path_split[-7:-4]
        first_part = ref_split[0]
        ref_name = f'{first_part[-3:]}_{ref_split[1]}_{ref_split[2]}'
        target_split = path_split[-3:]
        last_part = target_split[2]
        target_name = f'{target_split[0]}_{target_split[1]}_{last_part[:-4]}'

        if debug:
            print(f"Checking files {ref_name} vs. {target_name} for structure differences ({i}/{num_of_pairs})")
        pair_names = [ref_name, target_name]

        is_same, diff_regions = is_structure_same(warping_path=warping_path, pair_names=pair_names,
                                                  output_path=output_path, plot_path=plot_path,
                                                  segmentdivider=segmentdivider,
                                                  diff_max=diff_max, diff_min=diff_min, area_diff=area_diff,
                                                  plotting=plotting, debug=debug)

        if debug:
            print(f'The files {ref_name} and {target_name} contain the same music structure: {is_same}')

        if diff_regions:
            for i in diff_regions:
                min_value_ref = min(diff_regions[i]['ref'])
                max_value_ref = max(diff_regions[i]['ref'])
                if debug:
                    print(f'Diff regions of ref rec.: {min_value_ref} to {max_value_ref}s')

            for i in diff_regions:
                min_value_target = min(diff_regions[i]['target'])
                max_value_target = max(diff_regions[i]['target'])
                if debug:
                    print(f'Diff regions of target rec.: {min_value_target} to {max_value_target}s')

        if not is_same:
            duplicate_pairs_list.append(target_name)

        diff_regions_list.append(diff_regions)
    return duplicate_pairs_list, diff_regions_list


def is_structure_same(warping_path, pair_names, output_path, plot_path,
                      diff_max=3, diff_min=0.13, area_diff=10, plotting=False, debug=False,
                      segmentdivider=25, feature_rate=50):
    warping_path = np.load(warping_path)
    is_same, diff_regions = verify_path_slope(warping_path=warping_path, output_path=output_path, plot_path=plot_path,
                                              segmentdivider=segmentdivider,
                                              pair_names=pair_names, diff_max=diff_max,
                                              area_diff=area_diff, diff_min=diff_min, feature_rate=feature_rate,
                                              plotting=plotting, debug=debug)

    return is_same, diff_regions


def verify_path_slope(warping_path, output_path, plot_path, segmentdivider, pair_names,
                      diff_max=3, area_diff=10, diff_min=0.13, feature_rate=50, plotting=True, debug=False):
    pathx = np.array(warping_path[0, :])
    pathy = np.array(warping_path[1, :])

    # determination of sample numbers for line approximation
    pathxminval = min(pathx)  # determination of min and max values (start and beginning of line on x axis)
    pathxmaxval = max(pathx)
    pathxvalrange = pathxmaxval - pathxminval

    # makes sure that the range is divisible by the segmentdivider value
    modulo = pathxvalrange % segmentdivider
    pathxvalrange = pathxvalrange - modulo

    refpoint1xval = int(
        pathxminval + (pathxvalrange / segmentdivider))  # determination of point x value for approximation
    refpoint2xval = int(pathxminval + (pathxvalrange / segmentdivider) * (segmentdivider - 1))
    refpoint1xpos = int(
        np.argwhere(pathx == refpoint1xval)[0])  # finds out positions of array pathx at which these points are located

    refpoint2xpos = int(np.argwhere(pathx == refpoint2xval)[0])

    if debug:
        print(f'Point x start: {refpoint1xval} and x end: {refpoint2xval}')

    refpointsx = np.array([pathx[refpoint1xpos], pathx[
        refpoint2xpos]])  # creates arrays with x and y coordinates in format suitable for np.polyfit
    refpointsy = np.array([pathy[refpoint1xpos], pathy[refpoint2xpos]])

    # Line approximation --------------------------------
    coefficients = np.polyfit(refpointsx, refpointsy, 1)
    polynomial = np.poly1d(coefficients)

    linex = np.arange(start=0, stop=len(pathx), step=1)
    liney = polynomial(linex)

    testpointsnum = int((refpoint2xval - refpoint1xval) / 100)  # one point in cca 2 seconds cuz 50 fps
    if debug:
        print(f'No of testpoints: {testpointsnum}')

    # verifies whether the path between the two points used to approximate the curve actually lies on the curve
    refpointsvaldiff = refpoint2xval - refpoint1xval

    ref_slope = compute_slope(refpoint1xval, refpoint1xpos, refpoint2xval, refpoint2xpos)
    if debug:
        print(f'Ref_slope: {ref_slope}')

    if testpointsnum > refpointsvaldiff:  # ensures that the number of test points does not exceed the number of defined points between the reference points except the reference points themselves
        testpointsnum = refpointsvaldiff - 1

    testpointstep = refpointsvaldiff / testpointsnum  # step size

    # Testing of points ---------------------------------
    testpointxshift = testpointstep / 2  # variable that ensures that the first test point is not at the point where the curve intersects path
    list_of_x_points = []
    list_of_y_points = []
    list_of_slopes = []
    wrong_x_points = []
    wrong_y_points = []

    if plotting:
        plt.figure()

    is_same = True
    for i in range(0, testpointsnum, 1):  # cycle iterating across individual testing points
        if list_of_x_points:
            previous_x_testpoint = list_of_x_points[-1]
            previous_y_testpoint = list_of_y_points[-1]

            testpointxval = refpoint1xval + i * testpointstep + testpointxshift  # finding the value for testing
            testpointnearestxval = find_nearest(pathx, testpointxval)  # finding the nearest value to the test value
            testpointxpos = np.argwhere(pathx == testpointnearestxval)[
                0]  # finding the first index of the path element, which is equal to the given test position (index of pathx does not always match the value !!)
            pathval = float(pathy[testpointxpos])  # finding the value that matches the index found in the previous step
            current_slope = compute_slope(previous_x_testpoint, previous_y_testpoint, testpointnearestxval, pathval)
            list_of_slopes.append(current_slope)
            list_of_x_points.append(testpointnearestxval)
            list_of_y_points.append(pathval)

            if plotting:
                if abs(current_slope - ref_slope) > diff_max or current_slope <= diff_min:
                    plt.plot(testpointnearestxval, pathval, '+', markersize=12, color='red')
                else:
                    plt.plot(testpointnearestxval, pathval, '+', markersize=12, color='green')

            if abs(current_slope - ref_slope) > diff_max or current_slope <= diff_min:
                is_same = False
                # need to check if this is correct
                if wrong_x_points and wrong_y_points:
                    if (testpointnearestxval / feature_rate) - (previous_x_testpoint / feature_rate) > area_diff \
                            or (pathval / feature_rate) - (previous_y_testpoint / feature_rate) > area_diff:
                        wrong_x_points.append(previous_x_testpoint / feature_rate)
                        wrong_y_points.append(previous_y_testpoint / feature_rate)
                        wrong_x_points.append(testpointnearestxval / feature_rate)
                        wrong_y_points.append(pathval / feature_rate)
                    else:
                        wrong_x_points.append(testpointnearestxval / feature_rate)
                        wrong_y_points.append(pathval / feature_rate)
                else:
                    wrong_x_points.append(previous_x_testpoint / feature_rate)
                    wrong_y_points.append(previous_y_testpoint / feature_rate)
                    wrong_x_points.append(testpointnearestxval / feature_rate)
                    wrong_y_points.append(pathval / feature_rate)

        else:
            testpointxval = refpoint1xval + i * testpointstep + testpointxshift  # finding the value for testing
            testpointnearestxval = find_nearest(pathx, testpointxval)  # finding the nearest value to the test value
            testpointxpos = np.argwhere(pathx == testpointnearestxval)[
                0]  # finding the first index of the path element, which is equal to the given test position (index of pathx does not always match the value !!)
            pathval = float(pathy[testpointxpos])  # finding the value that matches the index found in the previous step

            list_of_x_points.append(testpointnearestxval)
            list_of_y_points.append(pathval)

    result_max = max(list_of_slopes)
    result_min = min(list_of_slopes)

    if debug:
        print(f'Result_max: {result_max}')
        print(f'Result_min: {result_min}')

    # compute the areas of inaccurate synchronization
    ref_x_points_table = [True]
    diff_regions = {}
    if wrong_x_points:
        list_of_wrong_x = wrong_x_points[:-1]
        list_of_wrong_x_shift = wrong_x_points[1:]
        for wrong_x, wrong_x_shift in zip(list_of_wrong_x, list_of_wrong_x_shift):
            current_diff = wrong_x_shift - wrong_x
            if current_diff >= area_diff:
                ref_x_points_table.append(False)
            else:
                ref_x_points_table.append(True)

        if all(item for item in ref_x_points_table):
            diff_regions[0] = {}
            diff_regions[0]['ref'] = {}
            diff_regions[0]['target'] = {}
            diff_regions[0]['ref'] = [min(wrong_x_points), max(wrong_x_points)]
            diff_regions[0]['target'] = [min(wrong_y_points), max(wrong_y_points)]

        else:
            num_of_operations = ref_x_points_table.count(False)
            idx_of_operation = [index for (index, item) in enumerate(ref_x_points_table) if not item]
            idx_of_operation.append(len(ref_x_points_table) - 1)

            keys = range(num_of_operations + 1)
            for i, idx in zip(keys, idx_of_operation):
                diff_regions[i] = {}
                diff_regions[i]['ref'] = {}
                diff_regions[i]['target'] = {}
                if i == 0:
                    diff_regions[i]['ref'] = wrong_x_points[:idx]
                    diff_regions[i]['target'] = wrong_y_points[:idx]
                elif idx == (len(ref_x_points_table) - 1):
                    prev_idx = idx_of_operation[i - 1]
                    diff_regions[i]['ref'] = wrong_x_points[prev_idx:idx + 1]
                    diff_regions[i]['target'] = wrong_y_points[prev_idx:idx + 1]
                else:
                    prev_idx = idx_of_operation[i - 1]
                    diff_regions[i]['ref'] = wrong_x_points[prev_idx:idx]
                    diff_regions[i]['target'] = wrong_x_points[prev_idx:idx]

    # Evaluation, if two recordings have the same structure---------
    # line_diff_max = abs(result_max - ref_slope)
    # line_diff_min = abs(result_min)
    # if line_diff_max > diff_max or line_diff_min <= diff_min:
    #     is_same = False
    # else:
    #     is_same = True

    if plotting:
        if not os.path.exists(f'{output_path}/plots/{plot_path}'):
            os.makedirs(f'{output_path}/plots/{plot_path}')
        plt.plot(pathx, pathy, color="black")
        plt.title(f'{pair_names[0]}, {pair_names[1]}')
        plt.plot(linex[refpoint1xval:refpoint2xval + 1], liney[refpoint1xval:refpoint2xval + 1])
        plt.plot(refpointsx, refpointsy, 'o', markersize=8, color='blue')
        plt.xlabel('reference recording')
        plt.ylabel('target recording')
        # tikzplotlib.save("wp_example.tex")
        plt.savefig(f'{output_path}/plots/{plot_path}/{pair_names[0]}_vs_{pair_names[1]}_wp.pdf', bbox_inches='tight')
        # plt.show()

    return is_same, diff_regions


# Helper functions ----------------------------------------------------------------------------------------------

# function for computing the slope of two points
def compute_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


# function for creating the output files
def create_output_file(diff_structure_list, path_to_output):
    if os.path.exists(path_to_output):
        os.remove(path_to_output)

    df = pd.DataFrame(diff_structure_list)
    writer = pd.ExcelWriter(path_to_output + '.xlsx')
    df.to_excel(writer, sheet_name='structure_diff')
    writer.save()

    df.to_csv(path_to_output + '.csv')


# function that returns nearest value to the input value from array
def find_nearest(array, value):
    index = np.abs(array - value).argmin()
    return array.flat[index]
