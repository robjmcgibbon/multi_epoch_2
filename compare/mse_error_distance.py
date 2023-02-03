import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
main_config = helpers.Config('config.yaml')
secondary_configs = main_config.load_secondary_configs('config.yaml')
configs = [main_config] + secondary_configs
config_names = [config.name for config in configs]
config_snapshots = [
    config.get_standard_spacing()
    # config.get_standard_spacing_one_snapshot_early()
    # config.get_tight_spacing()
    # config.get_every_snapshot()
    for config in configs
]
datasets = [config.load_data(snapshots) for config, snapshots in zip(configs, config_snapshots)]

for config in configs:
    assert config.redshift_to_predict == main_config.redshift_to_predict

output_features = {
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
}

dist = np.zeros((len(output_features), len(configs), len(configs)))
for i_output, (output_feature, input_properties) in enumerate(output_features.items()):
    log(f'Predicting {output_feature}')

    # Training regressor using data from i_config. Apply regressor to data from j_config
    for i_config in range(len(configs)):
        for j_config in range(len(configs)):
            log(f'i_config = {i_config+1}/{len(configs)}, j_config = {j_config+1}/{len(configs)}')
            config_1, data_1, snapshots_1 = configs[i_config], datasets[i_config], config_snapshots[i_config]
            config_2, data_2, snapshots_2 = configs[j_config], datasets[j_config], config_snapshots[j_config]

            score_ratios = np.zeros(main_config.n_dataset)
            for i_dataset in range(main_config.n_dataset):
                input_features_1 = [str(snap) + prop for snap in snapshots_1 for prop in input_properties]
                rf_train_1, rf_test_1 = config_1.generate_random_training_box(data_1, config_1.train_volume_fraction)

                rf_train_1 = rf_train_1[rf_train_1['clf_'+output_feature]]
                rf_test_1 = rf_test_1[rf_test_1['clf_'+output_feature]]
                y_train_1 = rf_train_1['regr_'+output_feature]
                y_test_1 = rf_test_1['regr_'+output_feature]
                X_train_1 = rf_train_1[input_features_1]
                X_test_1 = rf_test_1[input_features_1]

                regr = sklearn.ensemble.ExtraTreesRegressor(n_estimators=config_1.n_estimators,
                                                            n_jobs=config_1.n_process,
                                                            max_depth=config_1.get_max_depth(output_feature))
                regr.fit(X_train_1, y_train_1)

                y_pred_1 = regr.predict(X_test_1)
                score_1 = sklearn.metrics.mean_squared_error(y_test_1, y_pred_1)

                input_features_2 = [str(snap) + prop for snap in snapshots_2 for prop in input_properties]
                _, rf_test_2 = config_2.generate_random_training_box(data_2, config_2.train_volume_fraction)

                rf_test_2 = rf_test_2[rf_test_2['clf_'+output_feature]]
                y_test_2 = rf_test_2['regr_'+output_feature]
                X_test_2 = rf_test_2[input_features_2]

                y_pred_2 = regr.predict(X_test_2)
                score_2 = sklearn.metrics.mean_squared_error(y_test_2, y_pred_2)

                score_ratios[i_dataset] = score_2 / score_1

            dist[i_output, i_config, j_config] = np.mean(score_ratios)

    # Plotting distance heatmap for this output feature
    pd_dist = pd.DataFrame(dist[i_output], config_names, config_names)
    fig, ax = plt.subplots(1, figsize=(len(config_names), len(config_names)))
    seaborn.heatmap(pd_dist, square=True, cmap='Reds',
                    cbar_kws={'label': '$MSE_{apply} \quad / \quad MSE_{train}$'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'MSE errors when predicting {main_config.get_proper_name(output_feature, False)}')
    ax.set_xlabel('Application simulation')
    ax.set_ylabel('Training simulation')
    plot_name = f'btml_{output_feature}_mse_error_distance'
    plt.tight_layout()
    main_config.plot_show_save(plot_name, fig, force_save=True)

# Plotting distance heatmap of all output features
pd_dist = pd.DataFrame(np.sum(dist, axis=0), config_names, config_names)
fig, ax = plt.subplots(1, figsize=(len(config_names), len(config_names)))
seaborn.heatmap(pd_dist, square=True, cmap='Reds',
                cbar_kws={'label': '$MSE_{apply} \quad / \quad MSE_{train}$'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('Application simulation')
ax.set_ylabel('Training simulation')

plot_name = f'btml_all_mse_error_distance'
plt.tight_layout()
main_config.plot_show_save(plot_name, fig, force_save=False)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
