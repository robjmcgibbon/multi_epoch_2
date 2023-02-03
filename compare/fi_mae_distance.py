import datetime
import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

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

output_colors = [main_config.get_color(feat) for feat in output_features.keys()]
dist = np.zeros((len(output_features), len(configs), len(configs)))
for i_output, (output_feature, input_properties) in enumerate(output_features.items()):

    config_importances = []
    for i_config, (config, data, snapshots) in enumerate(zip(configs, datasets, config_snapshots)):
        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        mean_importance, _ = config.calculate_feature_importance(data, input_features, output_feature)
        config_importances.append(mean_importance)

    # Calculating distance between the feature importance vectors
    for i in range(len(configs)):
        for j in range(len(configs)):
            distance = np.sum(np.abs(config_importances[i] - config_importances[j]))
            dist[i_output, i, j] = distance
    dist[i_output] = np.tril(dist[i_output])

    # Plotting distance heatmap for feature importance vectors for this output feature
    pd_dist = pd.DataFrame(dist[i_output], config_names, config_names)
    fig, ax = plt.subplots(1, figsize=(len(config_names), len(config_names)))
    seaborn.heatmap(pd_dist, square=True, cmap='Reds',
                    mask=(pd_dist == 0), cbar_kws={'label': 'MAE between importance vectors'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'MAE when predicting {main_config.get_proper_name(output_feature, False)}')
    plot_name = f'btml_{output_feature}_mae_distance'
    plt.tight_layout()
    main_config.plot_show_save(plot_name, fig, force_save=False)

# Plotting distance heatmap for feature importance vectors of all output features
pd_dist = pd.DataFrame(np.sum(dist, axis=0), config_names, config_names)
fig, ax = plt.subplots(1, figsize=(len(config_names), len(config_names)))
seaborn.heatmap(pd_dist, square=True, cmap='Reds',
                mask=(pd_dist == 0), cbar_kws={'label': 'MAE between importance vectors'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

max_dist = np.argmax(dist, axis=0)
for i_y in range(1, len(config_names)):
    for i_x in range(0, i_y):
        maximum = max_dist[i_y, i_x]
        color = output_colors[maximum]
        ax.add_patch(patches.Rectangle((i_x+0.03, i_y+0.03), 0.94, 0.94, fill=False, edgecolor=color, lw=3))
for feat in output_features.keys():
    color = main_config.get_color(feat)
    name = main_config.get_proper_name(feat, False)
    ax.plot(0, 0, markersize=0, color=color, linewidth=5, label=name)
ax.legend(loc='upper right', title='Output feature with greatest MAE')

plot_name = f'btml_all_mae_distance'
plt.tight_layout()
main_config.plot_show_save(plot_name, fig, force_save=False)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
