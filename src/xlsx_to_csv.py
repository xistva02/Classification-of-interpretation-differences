import pandas as pd
from os.path import basename


convert_columns = {'Filename': str,
                   'Filename_No': str,
                   'label_CZ': str,
                   'label_random': str}

if __name__ == '__main__':

    from .config import *
    metadata_dir = f'../metadata'

    print(f'...processing file: {basename(__file__)}...')

    for i, user in enumerate(users):
        for session in sessions[i]:
            df_data = pd.read_excel(f'{metadata_dir}/_dataset_strings_{user}.xlsx',
                                    sheet_name=f'{session}', converters=convert_columns,
                                    engine='openpyxl')
            df_data.to_csv(f'{metadata_dir}/_dataset_strings_{user}_{session}.csv', index=False)
    print('---done---')