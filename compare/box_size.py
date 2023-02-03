import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')
ages = config.get_ages()

train_volume_fractions = [0.007, 0.03] + list(np.arange(0.1, 0.75, 0.2))

snapshots = config.get_standard_spacing()
# snapshots = config.get_standard_spacing_one_snapshot_early()
# snapshots = config.get_tight_spacing() # For this to work you will need to rerun preprocess data
# snapshots = config.get_every_snapshot()  # For this to work you will need to rerun preprocess data
snapshot_ages = [ages[snap] for snap in snapshots]
data = config.load_data(snapshots)

output_features = {
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
}

for i_output, (output_feature, input_properties) in enumerate(output_features.items()):

    n_plot = len(train_volume_fractions)
    fig, axs = plt.subplots(1, n_plot, figsize=(5*n_plot, 5))
    for i_plot, train_volume_fraction in enumerate(train_volume_fractions):

        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        fi = config.calculate_feature_importance(data, input_features, output_feature,
                                                 train_volume_fraction=train_volume_fraction)
        mean_importance, sem_importance = fi

        ax = axs[i_plot]
        for prop in input_properties:
            mean_values, sem_values = [], []
            for snap in snapshots:
                idx = input_features.index(str(snap)+prop)
                mean_values.append(mean_importance[idx])
                sem_values.append(sem_importance[idx])
            mean_values, sem_values = np.array(mean_values), np.array(sem_values)

            p = ax.plot(snapshot_ages, mean_values, '-o', label=config.get_proper_name(prop, False),
                        markersize=2, color=config.get_color(prop))
            ax.fill_between(snapshot_ages, mean_values-sem_values, mean_values+sem_values,
                            color=config.get_color(prop), edgecolor='w', alpha=0.2)
        ax.set_ylim(0)
        ax.set_ylabel(f'Feature importance')
        ax.legend(loc=2)

        padding = 0.015 * (np.max(snapshot_ages) - np.min(snapshot_ages))
        ax.set_xlim([np.min(snapshot_ages)-padding, np.max(snapshot_ages)+padding])
        xticks = np.linspace(np.min(snapshot_ages), np.max(snapshot_ages), 6)
        xticks = np.round(xticks, 1)
        ax.set_xticks(xticks)
        ax.set_xlabel('Universe age [Gyr]')

        config.add_redshift_labels(ax)

        train_box_length = np.cbrt(train_volume_fraction) * config.box_length
        ax.set_title(f'Box length: {train_box_length:.3g} Mpc')

    plt.tight_layout()
    plot_name = f'btml_{output_feature}_box_size'
    config.plot_show_save(plot_name, fig, force_save=True)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
