import datetime
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

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

cum_sfhs = []
for config, data, snapshots in zip(configs, datasets, config_snapshots):
    log(f'Plotting for {config.name}')
    snapshot_ages = [config.get_ages()[snap] for snap in snapshots]

    cum_sfh = np.zeros((data.shape[0], snapshots.shape[0]))
    for i_snap, snap in enumerate(snapshots):
        cum_sfh[:, i_snap] = np.array(data[str(snap)+'stellar_mass'] / data[f'{np.max(snapshots)}stellar_mass'])

    if np.any(np.isnan(cum_sfh)):
        raise ValueError('NaN values found. Only use halos with nonzero stellar mass')

    above_one = np.any(cum_sfh > 1, axis=1)
    log(f'Fraction of galaxies which have a cumulative SFH value greater than 1: {np.sum(above_one)/cum_sfh.shape[0]}')

    fig, ax = plt.subplots(1)
    for i_snap in range(1, len(snapshots)-1, 2):
        ax.hist(cum_sfh[:, i_snap], bins=np.arange(0, 1.25, 0.05), density=True,
                histtype='step', label=f't={snapshot_ages[i_snap]:.2g}Gyr')
    ax.set_xlim(0, 1.2)
    ax.legend()
    ax.set_xlabel('Stellar mass (t) / Final stellar mass')
    ax.set_ylabel('Density')
    main_config.plot_show_save(f'btml_{config.name}_sfh_dist', fig, force_save=False)

    cum_sfhs.append(cum_sfh)

fig, ax = plt.subplots(1)
for config, cum_sfh, snapshots in zip(configs, cum_sfhs, config_snapshots):
    snapshot_ages = [config.get_ages()[snap] for snap in snapshots]

    # mean_cum_sfh = np.mean(cum_sfh, axis=0)
    # std_cum_sfh = np.std(cum_sfh, axis=0)
    # p = ax.plot(snapshot_ages, mean_cum_sfh, '-', label=config.name)
    # ax.plot(snapshot_ages, mean_cum_sfh-std_cum_sfh, '--', color=p[0].get_color())
    # ax.plot(snapshot_ages, mean_cum_sfh+std_cum_sfh, '--', color=p[0].get_color())

    p = ax.plot(snapshot_ages, np.median(cum_sfh, axis=0), '-', label=config.name)
    ax.plot(snapshot_ages, np.quantile(cum_sfh, 0.25, axis=0), '--', color=p[0].get_color(), alpha=0.5)
    ax.plot(snapshot_ages, np.quantile(cum_sfh, 0.75, axis=0), '--', color=p[0].get_color(), alpha=0.5)
    # ax.fill_between(snapshot_ages, np.quantile(cum_sfh, 0.25, axis=0), np.quantile(cum_sfh, 0.75, axis=0),
    #                 color=p[0].get_color(), alpha=0.2)

linetypes = [mlines.Line2D([], [], color='k', linestyle='-'), mlines.Line2D([], [], color='k', linestyle='--')]
linetype_legend = ax.legend(linetypes, ['Median', 'Upper/Lower quartile'], loc='lower right')
ax.add_artist(linetype_legend)

ax.set_ylabel('Stellar mass (t) / Final stellar mass')
ax.set_xlabel('Universe age [Gyr]')
ax.set_ylim(-0.02)
ax.legend(loc='upper left')
main_config.plot_show_save('btml_compare_sfh', fig, force_save=False)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
