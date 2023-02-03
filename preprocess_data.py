import datetime

import numpy as np
import pandas as pd

import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')

generated_data_dir = config.get_generated_data_dir()

log('Loading histories')
histories = pd.read_pickle(generated_data_dir+'histories.pickle')

input_properties = [
    'bh_mass',
    # 'bh_dot',
    # 'cold_gas',
    # 'dm_fof_mass',
    'dm_sub_mass',
    'gas_mass',
    # 'hot_gas',
    # 'sfr',
    # 'log_sfr',
    # 'ssfr',
    # 'log_ssfr',
    'stellar_mass',
    # 'merge_bh_mass',
    # 'merge_dm_fof_mass',
    # 'merge_dm_sub_mass',
    # 'merge_gas_mass',
    # 'merge_stellar_mass'
]
output_features = ['bh_mass', 'gas_mass', 'sfr', 'stellar_mass', 'stellar_metallicity']

assert np.abs(np.min(histories['x']) / config.box_length) < 0.1
assert np.abs((np.max(histories['x']) / config.box_length) - 1) < 0.1

for output_feature in output_features:
    log(f'Creating {output_feature} clf columns')
    nonzero = (histories[output_feature] < 0) if 'mock' in output_feature else (histories[output_feature] != 0)
    histories['clf_'+output_feature] = nonzero
    log(f'Fraction of subhalos with nonzero {output_feature}: {np.sum(nonzero)/nonzero.shape[0]:.3g}')
    log(f'Number of subhalos with nonzero {output_feature}: {np.sum(nonzero)}')

    y = histories[output_feature].copy()
    if 'mock' not in output_feature:
        y[nonzero] = np.log10(y[nonzero])
    histories['regr_'+output_feature] = y

log(f'Filling missing input values')
max_snap = config.snapshot_to_predict
min_snap = config.get_min_snap_to_extract()
snapshots = list(range(min_snap+1, max_snap))
for snap in snapshots:
    for input_property in input_properties:
        if 'merge' in input_property:
            continue
        missing_values = (histories[str(snap)+input_property] == 0) & (histories[str(snap-1)+input_property] != 0)
        histories[str(snap)+input_property] += histories[str(snap+1)+input_property] * missing_values

log(f'Calculating log_sfr, log_ssfr for all snapshots')
for snap in snapshots:
    sfr = np.array(histories[str(snap) + 'sfr'])
    stellar_mass = np.array(histories[str(snap) + 'stellar_mass'])
    mask = (sfr != 0) & (stellar_mass != 0)

    log_sfr = np.zeros_like(sfr)
    log_sfr[mask] = np.log10(sfr[mask])
    histories[str(snap) + 'log_sfr'] = log_sfr

    log_ssfr = np.zeros_like(sfr)
    log_ssfr[mask] = np.log10(sfr[mask]) - np.log10(stellar_mass[mask])
    histories[str(snap) + 'log_ssfr'] = log_ssfr

# Only one of these should be uncommented, otherwise merger summing will be broken
snapshots = config.get_standard_spacing()
# snapshots = config.get_snapshots_within_n_gyr(2)
# snapshots = config.get_standard_spacing_one_snapshot_early()
# snapshots = config.get_tight_spacing()
# snapshots = config.get_every_snapshot()

log(f'Summing up mergers that occured between chosen snapshots')
merge_snapshots = ([min_snap] + list(snapshots))[:-1]
# The order I iterate through snapshots has to be reversed to avoid overwiting data that is needed later
for snap_1, snap_2 in zip(merge_snapshots[::-1], snapshots[::-1]):
    for input_property in input_properties:
        if 'merge' not in input_property:
            continue
        cols = [str(snap)+input_property for snap in range(snap_1, snap_2)]
        vals = np.sum(histories[cols], axis=1)
        histories[str(snap_2)+input_property] = vals

log(f'Defining columns used for cuts')
cut_columns = ['bh_mass', 'stellar_mass', 'ssfr', 'central', 'fof_mass', 'lowest_snap',
               'mock_g', 'mock_r', 'n_minor_merge', 'n_major_merge']
ssfr = np.zeros(histories.shape[0])
nonzero = histories['stellar_mass'] != 0
ssfr[nonzero] = histories['sfr'][nonzero] / histories['stellar_mass'][nonzero]
histories['ssfr'] = ssfr
histories['fof_mass'] = histories[str(config.snapshot_to_predict)+'dm_fof_mass']
histories['n_minor_merge'] = np.zeros(shape=histories.shape[0], dtype='int32')
histories['n_major_merge'] = np.zeros(shape=histories.shape[0], dtype='int32')
for snap in range(min_snap, np.max(snapshots)):
    histories['n_minor_merge'] += histories[str(snap)+'n_minor_merge']
    histories['n_major_merge'] += histories[str(snap)+'n_major_merge']
if 'p_late' in histories.columns:
    cut_columns.append('p_late')

log(f'Creating new dataframe')
input_columns = [str(snap)+prop for snap in snapshots for prop in input_properties]
regr_columns = ['regr_'+feat for feat in output_features]
clf_columns = ['clf_'+feat for feat in output_features]
output_columns = clf_columns + regr_columns + cut_columns + ['x', 'y', 'z']
data = histories[input_columns + output_columns]

log(f'Saving data')
data.to_pickle(generated_data_dir+'data.pickle')

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
