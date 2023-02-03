import datetime

import matplotlib.pyplot as plt
import numpy as np

import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')
ages = config.get_ages()
redshifts = config.get_redshifts()

snapshots = config.get_standard_spacing()
# snapshots = config.get_snapshots_within_n_gyr(3)
# snapshots = config.get_standard_spacing_one_snapshot_early()
# snapshots = config.get_tight_spacing()
# snapshots = config.get_every_snapshot()
snapshot_ages = [ages[snap] for snap in snapshots]

data = config.load_data(snapshots)

output_features = {
    'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass'],
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    # 'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass',
    #     'merge_gas_mass', 'merge_dm_sub_mass', 'merge_stellar_mass'],
    # 'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_stellar_mass'],
    # 'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass', 'merge_stellar_mass'],
    # 'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass'],
    # 'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass', 'merge_stellar_mass'],
}

plot_cumulative = False
plot_even_spaced_redshift = False
plot_single = False

mean_importances = {}
for output_feature, input_properties in output_features.items():
    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]

    try:
        mean_importance, sem_importance = config.calculate_feature_importance(data, input_features, output_feature)
    except helpers.NoDataError:
        log('No halos available for training')
        continue

    fig, ax = plt.subplots(1)
    integrals = {}
    plot_data = {}
    for input_property in input_properties:
        mean_values, sem_values = [], []
        for snap in snapshots:
            idx = input_features.index(str(snap)+input_property)
            mean_values.append(mean_importance[idx])
            sem_values.append(sem_importance[idx])
        mean_values, sem_values = np.array(mean_values), np.array(sem_values)
        integrals[input_property] = np.sum(mean_values)

        ax.plot(snapshot_ages, mean_values, 'o',
                linestyle='dashed' if 'merge' in input_property else 'solid',
                label=config.get_proper_name(input_property, False),
                markersize=2, color=config.get_color(input_property))

        if not plot_single:
            ax.fill_between(snapshot_ages, mean_values-sem_values, mean_values+sem_values,
                            color=config.get_color(input_property), edgecolor='w', alpha=0.2)
            plot_data[input_property] = [snapshot_ages, mean_values.tolist(), sem_values.tolist()]
        else:
            log(f'Calculating feature importance using only {input_property} as input')
            single_features = [str(snap) + input_property for snap in snapshots]
            single_mean_importance, _ = config.calculate_feature_importance(data, single_features, output_feature)

            single_mean_importance *= integrals[input_property]
            ax.plot(snapshot_ages, single_mean_importance, ':o',
                    markersize=2, color=config.get_color(input_property))

    if plot_single:
        # Plot unviewable points in order to add label to legend
        ax.plot(snapshot_ages, -np.ones_like(snapshots), '-ok', markersize=2, label='All input features used')
        ax.plot(snapshot_ages, -np.ones_like(snapshots), ':ok', markersize=2, label='Single input feature used')

    ax.set_ylim(0)
    ax.set_ylabel(f'Feature importance')
    ax.legend(ncol=2, fontsize=7, loc='upper right', handlelength=3)
    # ax.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')

# TODO: Still not even for different runs
    padding = 0.015 * (np.max(snapshot_ages) - np.min(snapshot_ages))
    ax.set_xlim([np.min(snapshot_ages)-padding, np.max(snapshot_ages)+padding])
    xticks = np.linspace(np.min(snapshot_ages), np.max(snapshot_ages), 6)
    xticks = np.round(xticks, 1)
    ax.set_xticks(xticks)
    ax.set_xlabel('Universe age [Gyr]')

    ax_labels = config.add_redshift_labels(ax)
    plot_data['ax_labels'] = ax_labels

    plot_title = f'Predict z={config.redshift_to_predict} {config.get_proper_name(output_feature, False)}'
    plot_title += f', {config.mask_desc}'
    ax.set_title(plot_title)

    plt.tight_layout()
    plot_name = f'btml_{output_feature}_feature_importance'
    config.plot_show_save(plot_name, fig, force_save=False, data=plot_data)

    if not plot_even_spaced_redshift:
        continue

    fig, ax = plt.subplots(1)
    snapshot_redshifts = [redshifts[s] for s in snapshots]
    for input_property in input_properties:
        mean_values, sem_values = [], []
        for snap in snapshots:
            idx = input_features.index(str(snap)+input_property)
            mean_values.append(mean_importance[idx])
            sem_values.append(sem_importance[idx])
        mean_values, sem_values = np.array(mean_values), np.array(sem_values)

        ax.plot(snapshot_redshifts, mean_values, '-o', label=config.get_proper_name(input_property, False),
                markersize=2, color=config.get_color(input_property))
        ax.fill_between(snapshot_redshifts, mean_values-sem_values, mean_values+sem_values,
                        color=config.get_color(input_property), edgecolor='w', alpha=0.2)
    ax.set_ylim(0)
    ax.set_ylabel(f'Feature importance')
    ax.legend(loc='upper right')

    ax.set_xticks(list(range(8)))
    ax.set_xlim([-0.1, 7.7])

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())

    ax2_xticks = []
    for X in ax.get_xticks():
        ax2_xticks.append(f'{config.get_lookback_time(int(X)):.3g}')
    ax.set_xlabel('z')
    ax2.set_xticklabels(ax2_xticks)
    ax2.set_xlabel('Lookback time [Gyr]')

    plt.tight_layout()
    plot_name = f'btml_{output_feature}_feature_importance_even_redshift'
    config.plot_show_save(plot_name, fig, force_save=False)

if plot_cumulative:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for output_feature, input_properties in output_features.items():
        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]

        mean_importance = mean_importances[output_feature]
        snapshot_importance = []
        for snap in snapshots:
            total = 0
            for input_property in input_properties:
                idx = input_features.index(str(snap)+input_property)
                total += mean_importance[idx]
            snapshot_importance.append(total)

        axs[0].plot(snapshot_ages, snapshot_importance, '-o',
                    label=config.get_proper_name(output_feature, False), markersize=2)
        axs[1].plot(snapshot_ages, np.cumsum(snapshot_importance), '-o',
                    label=config.get_proper_name(output_feature, False), markersize=2)

    axs[0].set_ylabel(f'Feature importance')
    axs[1].set_ylabel(f'Cumulative importance')

    for ax in axs:
        ax.set_ylim(0)
        ax.legend(loc=2)

        padding = 0.015 * (np.max(snapshot_ages) - np.min(snapshot_ages))
        ax.set_xlim([np.min(snapshot_ages)-padding, np.max(snapshot_ages)+padding])
        xticks = np.linspace(np.min(snapshot_ages), np.max(snapshot_ages), 6)
        xticks = np.round(xticks, 1)
        ax.set_xticks(xticks)
        ax.set_xlabel('Universe age [Gyr]')

        config.add_redshift_labels(ax)

    plt.tight_layout()
    plot_name = f'btml_feature_cumulative_importance'
    config.plot_show_save(plot_name, fig, force_save=False)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
