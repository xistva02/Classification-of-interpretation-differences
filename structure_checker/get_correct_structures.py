import numpy as np
import pandas as pd


if __name__ == '__main__':

    from config import *
    new_output_dir = f'same_structure/'

    for no_of_comp, user in enumerate(users):
        for session in sessions[no_of_comp]:

            path_to_metadata = f'../metadata/_dataset_strings_Dvorak_{session}.csv'

            for movement in movements:

                print(f'Sorting: {user}, {session}, mov{movement}')

                new_output_name = f'{user}_{session}_mov{movement}_same_structure'
                diff_structure_file = f'outputs/csv/{user}_{session}_mov{movement}_diff_structure.csv'

                df_diff = pd.read_csv(diff_structure_file)

                if df_diff.empty:
                    print(f'No data for {user}, {session}, mov{movement}')
                    print(df_diff)
                    continue

                df_diff['ID'] = df_diff['0'].astype(str).str[:3]
                diff_list = df_diff['ID'].to_list()

                df_all = pd.read_csv(path_to_metadata)
                df_all['ID'] = df_all['Filename'].astype(str).str[2:5]
                all_list = df_all['ID'].to_list()

                non_overlap = sorted(set(diff_list) ^ set(all_list))

                np.savetxt(f'{new_output_dir}/{user}_{session}_mov{movement}_same_structure.txt', non_overlap, fmt="%s")





