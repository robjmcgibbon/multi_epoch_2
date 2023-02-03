import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
# import umap

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

output_features = {
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
}

all_config_importances = [[] for config in configs]
for i_output, (output_feature, input_properties) in enumerate(output_features.items()):

    config_importances = []
    for i_config, (config, data, snapshots) in enumerate(zip(configs, datasets, config_snapshots)):
        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        mean_importance, _ = config.calculate_feature_importance(data, input_features, output_feature)
        config_importances.append(mean_importance)
        all_config_importances[i_config] = all_config_importances[i_config] + mean_importance.tolist()
    config_importances = np.array(config_importances)

    pca = sklearn.decomposition.PCA(n_components=2)
    embedding = pca.fit_transform(config_importances)
    fig, ax = plt.subplots(1)
    for i, name in enumerate(config_names):
        plt.scatter(embedding[i, 0], embedding[i, 1], label=name)
    ax.legend(ncol=2, loc='upper right', fontsize=5)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'PCA applied to FI vectors from predicting {output_feature}')
    plot_name = f'btml_{output_feature}_pca'
    main_config.plot_show_save(plot_name, fig, force_save=True)

    # TODO: How many configs are needed for this to work
    # TODO: umap for all output features combined
    # for i_umap in range(2):
    #     reducer = umap.UMAP()
    #     embedding = reducer.fit_transform(config_importances)
    #     fig, ax = plt.subplots(1)
    #     for i, name in enumerate(config_names):
    #         plt.scatter(embedding[i, 0], embedding[i, 1], label=name)
    #     ax.legend(ncol=2, loc='upper right', fontsize=5)
    #     ax.set_xlabel('Component 1')
    #     ax.set_ylabel('Component 2')
    #     ax.set_title(f'UMAP applied to FI vectors from predicting {output_feature}')
    #     plot_name = f'btml_{output_feature}_umap_{i_umap}'
    #     main_config.plot_show_save(plot_name, fig, force_save=False)

all_config_importances = [np.array(arr) for arr in all_config_importances]
pca = sklearn.decomposition.PCA(n_components=2)
embedding = pca.fit_transform(all_config_importances)
fig, ax = plt.subplots(1)
for i, name in enumerate(config_names):
    plt.scatter(embedding[i, 0], embedding[i, 1], label=name)
ax.legend(ncol=2, loc='upper right', fontsize=5)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
plot_name = f'btml_all_pca'
main_config.plot_show_save(plot_name, fig, force_save=True)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
