import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mrmr import mrmr_classif
import glob
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.metrics import f1_score


class Classification:
    def __init__(self, user, session, movement, scenario, label, annot_style='chroma',
                 label_path=None, mrmr_path=None, filenames=[]):
        self.user = user
        self.session = session
        self.movement = movement
        self.scenario = scenario
        self.label = label
        self.annot_style = annot_style  # chroma or beat, depends on the sync scenario (if act fun is used)
        self.filenames = filenames
        self.data_path = f'data/{self.user}/{self.session}/mov{self.movement}'
        self.path_to_same_structure = f'structure_checker/outputs/csv/{self.user}_{self.session}_mov{self.movement}_same_structure.csv'

        if label_path is None:
            self.label_path = f'metadata/_dataset_strings_{self.user}_{self.session}.csv'
        else:
            self.label_path = label.path
        if mrmr_path is None:
            self.mrmr_path = f'mrmr/{self.scenario}_{self.user}_{self.session}_{self.movement}_mrmr_features_{self.label}.txt'
        else:
            self.mrmr_path = mrmr_path

        self.create_dirs()

    def create_dirs(self):
        """
        Create all required dirs for given scenario: 'mrmr', 'results', and 'figs + 'data_matrices'.
        """
        dirs = ['mrmr', 'figs', 'results']
        for basic_dir_path in dirs:
            if not os.path.exists(basic_dir_path):
                os.makedirs(basic_dir_path)
        if not os.path.exists(f'data_matrices/{self.scenario}'):
            os.makedirs(f'data_matrices/{self.scenario}')

    def preprocess_datamatrix(self):

        data_path = f'data/{self.user}/{self.session}/mov{self.movement}/sync_measures/{self.annot_style}'
        sync_files = glob.glob(f'{data_path}/*.txt')
        all_sync_files = []
        for file in sync_files:
            filename = Path(file).stem
            all_sync_files.append(str(filename[:3]))

        sync_temp_list = list(zip(all_sync_files, sync_files))
        df_sync = pd.DataFrame(sync_temp_list, columns=['Filename', 'Path'], dtype='str')
        df_correct_files = pd.read_csv(self.path_to_same_structure, dtype='str')['Filename']
        df_joint = pd.merge(df_sync, df_correct_files, on=['Filename', 'Filename'], how='left', indicator='sync_well')
        df_joint['sync_well'] = np.where(df_joint.sync_well == 'both', True, False)
        df_selection = df_joint.loc[df_joint['sync_well'] == True]
        sync_final_list = df_selection['Path'].to_list()
        df_filenames_process = df_selection['Filename']

        sync_filenames = []
        data_nonscaled = []

        for file in sync_final_list:
            filename = Path(file).stem
            sync_filenames.append(filename)
            with open(file, 'r') as f:
                loaded_measures = [line.rstrip() for line in f]
            loaded_measures = list(map(float, loaded_measures))

            if self.scenario == 'movements':
                data_nonscaled.append(loaded_measures[-1] - loaded_measures[0])
            elif self.scenario == 'motifs':
                segmented_motifs = np.loadtxt(f'segmentation/motifs_{user}_{session}_mov{movement}.txt', dtype=int)
                segment_list = []
                last_values = []
                for i, item in enumerate(segmented_motifs):
                    if i == 0:
                        starting_item = loaded_measures[0]
                        last_item = loaded_measures[item]
                        last_values.append(last_item)
                        segment_list.append(last_item - starting_item)
                    else:
                        first_item = last_values[-1]
                        last_item = loaded_measures[item]
                        last_values.append(last_item)
                        segment_list.append(last_item - first_item)
                data_nonscaled.append(segment_list)

            elif self.scenario == 'measures':
                diff_time = [y - x for x, y in zip(loaded_measures[:-1], loaded_measures[1:])]
                data_nonscaled.append(diff_time)
            else:
                raise Exception('Wrong scenario selected.')

        if self.scenario == 'motifs':
            x_measures = np.arange(1, len(segmented_motifs) + 1, 1, dtype=int)
        else:
            x_measures = np.arange(1, len(loaded_measures), 1, dtype=int)

        scaler = StandardScaler(with_mean=True, with_std=True)

        if self.scenario == 'movements':
            columns = [f'mov{movement}']
            x = scaler.fit_transform(np.array(data_nonscaled).reshape(-1, 1))
        elif self.scenario == 'motifs':
            columns = ['motif' + str(x) for x in range(1, len(x_measures) + 1)]
            x = scaler.fit_transform(np.array(data_nonscaled))
        else:
            columns = ['measure' + str(x) for x in range(1, len(x_measures) + 1)]
            x = scaler.fit_transform(np.array(data_nonscaled))

        df_scaled = pd.DataFrame(x, index=sync_filenames, columns=columns)
        df_scaled.to_csv(f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{self.movement}_scaled.csv',
                         encoding='utf-8',
                         index=True)

        df_nonscaled = pd.DataFrame(np.array(data_nonscaled), index=sync_filenames, columns=columns)
        df_nonscaled.to_csv(
            f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{self.movement}_nonscaled.csv',
            encoding='utf-8',
            index=True)

        return df_filenames_process

    def compute_mrmr(self, df_data=None, df_labels=None, n_mrmr=10, save=True):
        # if df_labels is not provided, it selects the last column of df_data as df_labels

        if df_data is None or df_labels is None:
            raise Exception('Provide df_data and df_labels parameters!')

        if self.scenario == 'movements':
            prefix = 'mov'
        elif self.scenario == 'measures':
            prefix = 'measure'
        elif self.scenario == 'motifs':
            prefix = 'motif'
        else:
            raise Exception('Wrong scenario selected.')

        if not n_mrmr:
            features = mrmr_classif(df_data, df_labels, K=len(df_data.columns))
        else:
            features = mrmr_classif(df_data, df_labels, K=n_mrmr)

        if prefix:
            selected_features = [sub.replace(f'{prefix}', '') for sub in features]
            selected_features = [int(x) for x in selected_features]
        else:
            selected_features = [int(x) for x in features]

        if save:
            np.savetxt(self.mrmr_path,
                       selected_features, fmt='%i')

        return selected_features

    def pca_with_mrmr(self, df_scaled=None, df_nonscaled=None, df_labels=None, selected_features=None,
                      n_components=2, stats=False, plotting=True):
        if df_scaled is None or df_nonscaled is None:
            raise Exception('Provide df_scaled and df_nonscaled parameter!')

        if selected_features is None:
            selected_features = np.load(self.mrmr_path)
            selected_features = [x - 1 for x in selected_features]
        else:
            selected_features = [x - 1 for x in selected_features]

        df_selected = df_scaled.iloc[:, selected_features]
        x_selected = df_selected.values

        pca = PCA(n_components=n_components)
        pca.fit(x_selected)
        x_pca = pca.transform(x_selected)

        if plotting:
            plt.figure()
            plt.scatter(x_pca[:, 0], x_pca[:, 1], c=df_labels, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
            plt.xlabel("PCA component 1")
            plt.ylabel("PCA component 2")
            plt.savefig(f'figs/pca_{self.scenario}_{self.user}_{self.session}_mov{self.movement}_{self.label}.pdf',
                        bbox_inches='tight')
            plt.close()

        if stats:
            df_orig_data_subset = df_nonscaled.iloc[:, selected_features]
            df_1_temp = pd.concat([df_orig_data_subset, df_labels], axis=1)
            df_0 = df_1_temp[df_1_temp.iloc[:, -1] == 0]
            df_1 = df_1_temp[df_1_temp.iloc[:, -1] == 1]
            df_mean_0 = df_0.mean()
            df_std_0 = df_0.std()
            df_mean_1 = df_1.mean()
            df_std_1 = df_1.std()

            df_all = pd.concat([df_mean_1[:-1], df_mean_0[:-1], df_std_1[:-1], df_std_0[:-1]], axis=1)
            df_all.columns = ['1_mean', '0_mean', '1_std', '0_std']

            df_all.to_csv(
                f'results/stats_{self.scenario}_{self.user}_{self.session}_mov{self.movement}_{self.label}.csv',
                encoding='utf-8', index=False)

        return x_pca

    def get_labels(self, filenames):
        try:
            df_labels = pd.read_csv(f'{self.label_path}',
                                    converters={'Filename': str, 'label_CZ': str, 'Filename_No': str})
            if self.scenario == 'movements':
                filenames = [filename[:3] for filename in filenames]
                df_labels = df_labels.iloc[np.where(df_labels.Filename_No.isin(filenames))]
            else:
                filenames = filenames.to_list()
                df_labels = df_labels.iloc[np.where(df_labels.Filename_No.isin(filenames))]

            if self.label == 'label_gender':
                df_labels.loc[df_labels.label_gender == 'F', 'label_gender'] = 1.0
                df_labels.loc[df_labels.label_gender == 'M', 'label_gender'] = 0.0

            labels = df_labels[f'{self.label}'].astype(float).squeeze()

            return labels

        except Exception as e:
            print(f'Problem with label handling; {e}')

    def classification_svm(self, df_data, df_labels, num_of_it=1000, average_type='weighted',
                           method='NuSVM', mrmr=True, balance=True, save=True):

        fscore = []
        df_merged = pd.concat([df_data, df_labels], axis=1)

        for i in range(num_of_it):
            if balance:
                df_0_subset = df_merged[df_merged[f'{self.label}'] == 0]
                df_1_subset = df_merged[df_merged[f'{self.label}'] == 1]

                if len(df_0_subset.index) <= len(df_1_subset.index):
                    df_1_subset = df_1_subset.sample(n=len(df_0_subset.index))
                else:
                    df_0_subset = df_0_subset.sample(n=len(df_1_subset.index))

                df_new_allset = pd.concat([df_1_subset, df_0_subset])
                df_x = df_new_allset.iloc[:, :-1]
                labels = df_new_allset.iloc[:, -1].values
            else:
                df_x = df_merged.iloc[:, :-1]
                labels = df_merged.iloc[:, -1].values

            if mrmr:
                selected_features = np.loadtxt(self.mrmr_path,
                                               dtype=int)
                selected_features = [x - 1 for x in selected_features]
                x_selected = df_x.iloc[:, selected_features].values
            else:
                x_selected = df_x.values

            x_train, x_test, y_train, y_test = train_test_split(x_selected, labels, shuffle=True, test_size=0.25,
                                                                stratify=labels)

            if method == 'LinearSVM':
                svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                                multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None,
                                verbose=0, random_state=None, max_iter=1000)
            else:
                svc = NuSVC(gamma='auto')

            svc.fit(x_train, y_train)
            y_pred = svc.predict(x_test)

            fscore_val = f1_score(y_test, y_pred, average=average_type, labels=np.unique(y_pred))
            fscore.append(fscore_val)

        if save:
            with open(
                    f'results/fscore_pca_{self.scenario}_{self.user}_{self.session}_mov{self.movement}_{self.label}.txt',
                    'w') as f:
                f.write(str(np.mean(fscore)))

        return np.mean(fscore), np.std(fscore)

    def load_df(self, filenames):
        if self.scenario == 'movements':
            df_nonscaled = pd.read_csv(f'data_matrices/movements/{self.user}_{self.session}_nonscaled.csv')
            df_scaled = pd.read_csv(f'data_matrices/movements/{self.user}_{self.session}_scaled.csv').iloc[:, 1:]
            filenames = df_nonscaled.iloc[:, 0].values
        else:
            df_nonscaled = pd.read_csv(
                f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{self.movement}_nonscaled.csv')
            df_scaled = pd.read_csv(
                f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{self.movement}_scaled.csv').iloc[:, 1:]

        # self.filenames = df_nonscaled.iloc[:, 0].values
        df_nonscaled = df_nonscaled.iloc[:, 1:]
        df_labels = self.get_labels(filenames=filenames)
        df_bool = df_labels.notna().values

        df_nonscaled = df_nonscaled.loc[df_bool, :]
        df_scaled = df_scaled.loc[df_bool, :]
        df_labels.dropna(inplace=True)

        df_labels.index = range(1, len(df_labels) + 1)
        df_scaled.index = range(1, len(df_scaled.index) + 1)
        df_nonscaled.index = range(1, len(df_nonscaled.index) + 1)

        return df_nonscaled, df_scaled, df_labels

    def adjust_columns(self, df, mov):
        df.columns = ['CD', f'mov{mov}']
        return df.astype({'CD': str, f'mov{mov}': str})

    def load_and_add(self, movements=None):

        if movements is None:
            movements = ['1', '2', '3', '4']

        df_nonscaled_all = pd.read_csv(f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov1_nonscaled.csv')
        df_nonscaled_all = self.adjust_columns(df_nonscaled_all, mov='1')
        df_nonscaled_all = pd.DataFrame(df_nonscaled_all['CD'])

        df_scaled_all = pd.read_csv(f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov1_scaled.csv')
        df_scaled_all = self.adjust_columns(df_scaled_all, mov='1')
        df_scaled_all = pd.DataFrame(df_scaled_all['CD'])

        for movement in movements:
            path_nonscaled = f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{movement}_nonscaled.csv'
            path_scaled = f'data_matrices/{self.scenario}/{self.user}_{self.session}_mov{movement}_scaled.csv'

            df_nonscaled_temp = pd.read_csv(path_nonscaled)
            df_nonscaled_temp = self.adjust_columns(df_nonscaled_temp, mov=movement)

            df_scaled_temp = pd.read_csv(path_scaled)
            df_scaled_temp = self.adjust_columns(df_scaled_temp, mov=movement)

            df_nonscaled_all = df_nonscaled_all.join(df_nonscaled_temp[f'mov{movement}'])
            df_scaled_all = df_scaled_all.join(df_scaled_temp[f'mov{movement}'])
            # df_nonscaled_all[f'mov{movement}'] = df_nonscaled_temp[f'mov{movement}']
            # df_scaled_all[f'mov{movement}'] = df_scaled_temp[f'mov{movement}']

        for movement in movements:
            df_nonscaled_all[f'mov{movement}'].replace('', np.nan, inplace=True)
            df_nonscaled_all.dropna(subset=[f'mov{movement}'], inplace=True)
            df_scaled_all[f'mov{movement}'].replace('', np.nan, inplace=True)
            df_scaled_all.dropna(subset=[f'mov{movement}'], inplace=True)

        df_nonscaled_all.to_csv(f'data_matrices/movements/{self.user}_{self.session}_nonscaled.csv', index=False)
        df_scaled_all.to_csv(f'data_matrices/movements/{self.user}_{self.session}_scaled.csv', index=False)


def run_classification(user, session, movement, scenario, label, stats=False, save=True, debug=False):
    print(
        f'Classification of recordings; scenario: {scenario}, user: {user}, session: {session}, mov: {movement}, label: {label}')
    cl = Classification(user=user, session=session, movement=movement, scenario=scenario, label=label)
    ďf_filenames_process = cl.preprocess_datamatrix()

    if scenario == 'movements':
        print(f'Processing mov{movement} and waiting for all data.')

        if movement == '4':
            cl.load_and_add()
            df_nonscaled, df_scaled, df_labels = cl.load_df(filenames=ďf_filenames_process)
            features = cl.compute_mrmr(df_data=df_scaled, df_labels=df_labels, save=save)
            cl.pca_with_mrmr(df_scaled=df_scaled, df_nonscaled=df_nonscaled, df_labels=df_labels,
                             selected_features=features,
                             stats=stats)
        else:
            return 0, 0


    else:
        df_nonscaled, df_scaled, df_labels = cl.load_df(filenames=ďf_filenames_process)
        features = cl.compute_mrmr(df_data=df_scaled, df_labels=df_labels, save=save)
        cl.pca_with_mrmr(df_scaled=df_scaled, df_nonscaled=df_nonscaled, df_labels=df_labels,
                         selected_features=features,
                         stats=stats)

    try:
        fscore, std = cl.classification_svm(df_data=df_scaled, df_labels=df_labels, save=save)
    except Exception as e:
        fscore = 0
        std = 0
        if debug:
            logger.warning(
                f'Cannot process the parameters; user: {user}, session: {session}, mov: {movement}, label: {label}')
            logger.exception(e, exc_info=True)
        else:
            print(f'Cannot use SVM on provided data; user: {user}, session: {session}, mov: {movement}, label: {label}')

    return fscore, std


if __name__ == '__main__':

    from src.config import *

    print(f'---processing {os.path.basename(__file__)}---')

    fscores = {}
    stds = {}
    debug = True

    if debug:
        logging.basicConfig(filename='strings_analysis.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info('strings_analysis logger')
        logging.getLogger('strings_analysis').addHandler(logging.StreamHandler())
        logging.getLogger('matplotlib.font_manager').disabled = True
        logger = logging.getLogger('strings_analysis')

    for scenario in scenarios:
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

                    for label in labels:
                        fscore, std = run_classification(user=user,
                                                         session=session,
                                                         movement=movement,
                                                         scenario=scenario,
                                                         label=label, stats=True, save=True, debug=debug)
                        print(f'Fscore: {fscore}, std: {std}')
                        fscores[f'{scenario}_{user}_{session}_mov{movement}_{label}'] = fscore
                        stds[f'{scenario}_{user}_{session}_mov{movement}_{label}'] = std

    all_results = []
    for (fscore_key, fscore_item), (std_key, std_item) in zip(fscores.items(), stds.items()):
        all_results.append([fscore_key, round(fscore_item, 3), round(std_item, 3)])
    
    fscore_df = pd.DataFrame(all_results, columns=['run', 'fscore', 'std'])
    fscore_df.to_excel(f'classification_fscore.xlsx', index=False)
