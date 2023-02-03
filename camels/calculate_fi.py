import os
import sys

import numpy as np
import pandas as pd
import sklearn.ensemble
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

base_dir = helpers.Config.get_base_dir()

snapshots = np.arange(6, 34, 3)

for sim in ['IllustrisTNG', 'SIMBA']:
    log(f'Running for {sim}')
    data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/'
    run_names = sorted(os.listdir(data_dir))
    for run_name in run_names:
        log(f'Calculating feature importance for {run_name}')
        # TODO: Cut data so only tracked halos are used
        histories = pd.read_pickle(f'{data_dir}{run_name}/histories.pickle')
        histories = histories[histories[str(min(snapshots))+'stellar_mass'] != 0]

        input_properties = ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass']
        output_features = ['bh_mass', 'gas_mass', 'sfr', 'stellar_mass', 'stellar_metallicity']

        hubble_constant = 0.6711
        box_length = 25 / hubble_constant

        # TODO: Set value for stellar mass
        mask = histories['stellar_mass'] > 10**9
        histories = histories[mask]
        log(f'There are {histories.shape[0]} halos with stellar mass > 10**9')
        min_snap = np.min(snapshots)
        n_trackable = np.sum(histories[str(min_snap)+'dm_sub_mass'] != 0)
        log(f'There are {n_trackable} halos which can be traced to snapshot {min_snap}')

        for output_feature in ['bh_mass', 'gas_mass', 'sfr', 'stellar_metallicity']:
            arr = histories[output_feature]
            log(f'Fraction {output_feature} equal to zero: {np.sum(arr==0)/arr.shape[0]:.3g}')
            min_nonzero = np.min(arr[arr != 0])
            histories[output_feature] = np.maximum(arr, min_nonzero*np.ones_like(arr))

        for output_feature in output_features:
            histories['regr_'+output_feature] = np.log10(histories[output_feature])

        input_columns = [str(snap)+prop for snap in snapshots for prop in input_properties]
        regr_columns = ['regr_'+feat for feat in output_features]
        output_columns = regr_columns + ['x', 'y', 'z']
        data = histories[input_columns + output_columns]

        output_features = {
            'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass'],
            'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
            'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
            'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
            'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass']
        }
        feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
        if not os.path.exists(feature_importance_dir):
            os.makedirs(feature_importance_dir)
        with open(feature_importance_dir+'output_features.yaml', 'w') as yaml_file:
            yaml.dump(output_features, yaml_file)
        np.save(feature_importance_dir+'snapshots.npy', snapshots)

        for output_feature, input_properties in output_features.items():
            input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]

            n_dataset = 10
            n_estimators = 100
            n_process = 5
            max_depth = 12
            frac = 0.7

            importances = np.zeros((n_dataset, len(input_features)))
            for i_dataset in range(n_dataset):

                train_box_length = np.cbrt(frac) * box_length

                shift = np.random.uniform(low=0, high=box_length, size=3)
                pos = np.array(data[['x', 'y', 'z']])
                pos += shift
                pos %= box_length

                train_mask = np.all(pos < train_box_length, axis=1)
                train_data = data[train_mask]

                X_train = train_data[input_features]
                y_train = train_data['regr_'+output_feature]

                regr = sklearn.ensemble.ExtraTreesRegressor(n_estimators=n_estimators,
                                                            n_jobs=n_process,
                                                            max_depth=max_depth)
                regr.fit(X_train, y_train)

                importances[i_dataset] = regr.feature_importances_

            mean_importance = np.mean(importances, axis=0)
            sem_importance = 1.96 * np.std(importances, axis=0) / np.sqrt(n_dataset)

            np.save(f'{feature_importance_dir}{output_feature}_mean_importance.npy', mean_importance)
            np.save(f'{feature_importance_dir}{output_feature}_sem_importance.npy', sem_importance)

log('Script finished')
