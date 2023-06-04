import pandas as pd
from config import *

convert_columns = {'Filename': str,
                   'label_CZ': str,
                   'Filename_No': str}

metadata_dir = f'../metadata'

for i, user in enumerate(users):
    for session in sessions[i]:
        df_data = pd.read_excel(f'{metadata_dir}/_dataset_strings_{user}.xlsx',
                                sheet_name=f'{session}', converters=convert_columns,
                                engine='openpyxl')
        df_data.to_csv(f'{metadata_dir}/_dataset_strings_{user}_{session}.csv', index=False)
