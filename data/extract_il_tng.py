import datetime
import os
import multiprocessing
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')

assert config.simulation in ['tng', 'illustris']

raw_data_dir = config.get_raw_data_dir()
generated_data_dir = config.get_generated_data_dir()

ages = helpers.get_tng_ages() if config.simulation == 'tng' else helpers.get_illustris_ages()
redshifts = helpers.get_tng_redshifts() if config.simulation == 'tng' else helpers.get_illustris_redshifts()


def lhalotree_extract_histories_from_file(script_config, filename):
    log(f'Starting processing file: {filename}')

    with h5py.File(filename, 'r') as file:
        n_halos_in_tree = np.array(file['/Header/TreeNHalos'])

        histories = []
        for i_tree, n_halo in enumerate(n_halos_in_tree):
            arr = {}
            tree = file[f'Tree{i_tree}']

            # Convert positions from kpc to Mpc
            arr_loc = np.array(tree['SubhaloPos']) / 1000
            arr['x'] = arr_loc[:, 0] / script_config.get_hubble_constant()
            arr['y'] = arr_loc[:, 1] / script_config.get_hubble_constant()
            arr['z'] = arr_loc[:, 2] / script_config.get_hubble_constant()

            # Convert mass to solar units
            arr_mass_type = np.array(tree['SubhaloMassType']) * (10 ** 10) / script_config.get_hubble_constant()
            arr['bh_mass'] = np.array(tree['SubhaloBHMass']) * (10 ** 10) / script_config.get_hubble_constant()
            arr['bh_dot'] = np.array(tree['SubhaloBHMdot']) * 10.22  # Convert to solar masses per year
            arr['dm_fof_mass'] = np.array(tree['Group_M_Crit200']) * (10 ** 10) / script_config.get_hubble_constant()
            arr['dm_sub_mass'] = arr_mass_type[:, 1]
            arr['gas_mass'] = arr_mass_type[:, 0]
            arr['mock_g'] = np.array(tree['SubhaloStellarPhotometrics'])[:, 4]
            arr['mock_k'] = np.array(tree['SubhaloStellarPhotometrics'])[:, 3]
            arr['mock_r'] = np.array(tree['SubhaloStellarPhotometrics'])[:, 5]
            arr['mock_u'] = np.array(tree['SubhaloStellarPhotometrics'])[:, 0]
            arr['sfr'] = np.array(tree['SubhaloSFR'])
            arr['stellar_mass'] = arr_mass_type[:, 4]
            arr['stellar_metallicity'] = np.array(tree['SubhaloStarMetallicity'])

            arr['main_prog_index'] = np.array(tree['FirstProgenitor'])
            arr['next_prog_index'] = np.array(tree['NextProgenitor'])
            arr['snap_num'] = np.array(tree['SnapNum'])
            arr['subhalo_id'] = np.array(tree['SubhaloNumber'])

            arr_central_index = np.array(tree['FirstHaloInFOFGroup'])
            arr['is_central'] = np.zeros(n_halo, dtype=bool)
            for i_halo, i_central in enumerate(arr_central_index):
                arr['dm_fof_mass'][i_halo] = arr['dm_fof_mass'][i_central]
                arr['is_central'][i_halo] = (i_halo == i_central)

            tree_histories = helpers.fill_histories_dataframe(script_config, arr)
            histories.append(tree_histories)

    histories = pd.concat(histories)
    log(f'Finished processing file: {filename}')
    return histories


def sublink_extract_histories_from_file(script_config, filename):
    log(f'Starting processing file: {filename}')

    with h5py.File(filename, 'r') as file:
        # Note that there is a difference between SubhaloID and SubfindID
        arr = {'subhalo_id': np.array(file['SubfindID'])}
        arr_sublink_id = np.array(file['SubhaloID'])
        dict_index_from_sublink_id = {sub_id: i_arr for i_arr, sub_id in enumerate(arr_sublink_id)}
        n_halo = arr['subhalo_id'].shape[0]

        # Convert positions from kpc to Mpc
        arr_loc = np.array(file['SubhaloPos']) / 1000
        arr['x'] = arr_loc[:, 0] / script_config.get_hubble_constant()
        arr['y'] = arr_loc[:, 1] / script_config.get_hubble_constant()
        arr['z'] = arr_loc[:, 2] / script_config.get_hubble_constant()

        # Convert mass to solar units
        arr_mass_type = np.array(file['SubhaloMassType']) * (10 ** 10) / script_config.get_hubble_constant()
        arr['bh_mass'] = np.array(file['SubhaloBHMass']) * (10 ** 10) / script_config.get_hubble_constant()
        arr['bh_dot'] = np.array(file['SubhaloBHMdot']) * 10.22  # Convert to solar masses per year
        arr['dm_fof_mass'] = np.array(file['Group_M_Crit200']) * (10 ** 10) / script_config.get_hubble_constant()
        arr['dm_sub_mass'] = arr_mass_type[:, 1]
        arr['gas_mass'] = arr_mass_type[:, 0]
        arr['mock_g'] = np.array(file['SubhaloStellarPhotometrics'])[:, 4]
        arr['mock_k'] = np.array(file['SubhaloStellarPhotometrics'])[:, 3]
        arr['mock_r'] = np.array(file['SubhaloStellarPhotometrics'])[:, 5]
        arr['mock_u'] = np.array(file['SubhaloStellarPhotometrics'])[:, 0]
        arr['sfr'] = np.array(file['SubhaloSFR'])
        arr['stellar_mass'] = arr_mass_type[:, 4]
        arr['stellar_metallicity'] = np.array(file['SubhaloStarMetallicity'])

        arr_central_sublink_id = np.array(file['FirstSubhaloInFOFGroupID'])
        arr_main_prog_sublink_id = np.array(file['FirstProgenitorID'])
        arr_next_prog_sublink_id = np.array(file['NextProgenitorID'])
        arr_central_index = np.zeros(n_halo, dtype='int64')
        arr['main_prog_index'] = -1 * np.ones(n_halo, dtype='int64')
        arr['next_prog_index'] = -1 * np.ones(n_halo, dtype='int64')
        for i_halo in range(n_halo):
            central_sublink_id = arr_central_sublink_id[i_halo]
            arr_central_index[i_halo] = dict_index_from_sublink_id[central_sublink_id]

            main_prog_sublink_id = arr_main_prog_sublink_id[i_halo]
            if main_prog_sublink_id != -1:
                arr['main_prog_index'][i_halo] = dict_index_from_sublink_id[main_prog_sublink_id]

            next_prog_sublink_id = arr_next_prog_sublink_id[i_halo]
            if next_prog_sublink_id != -1:
                arr['next_prog_index'][i_halo] = dict_index_from_sublink_id[next_prog_sublink_id]

        arr['is_central'] = np.zeros(n_halo, dtype=bool)
        for i_halo, i_central in enumerate(arr_central_index):
            arr['dm_fof_mass'][i_halo] = arr['dm_fof_mass'][i_central]
            arr['is_central'][i_halo] = (i_halo == i_central)

        arr['snap_num'] = np.array(file['SnapNum'])

    histories = helpers.fill_histories_dataframe(script_config, arr)
    log(f'Finished processing file: {filename}')
    return histories


def add_morphologies(histories, script_config):
    morphology_filename = raw_data_dir + '../../' + 'morphologies_deeplearn.hdf5'
    if os.path.exists(morphology_filename) and script_config.snapshot_to_predict == 99:
        log('Extracting galaxy morphologies')
        with h5py.File(morphology_filename, 'r') as file:
            arr_subhalo_id = np.array(file['/Snapshot_99/SubhaloID'])
            p_late = np.array(file['/Snapshot_99/P_Late'])
            p_s_zero = np.array(file['/Snapshot_99/P_S0'])
            p_s_zero[p_late > 0.5] = -1
            p_sab = np.array(file['/Snapshot_99/P_Sab'])
            p_sab[p_late < 0.5] = -1
        morphology = {}
        for i_halo, subhalo_id in enumerate(arr_subhalo_id):
            morphology[subhalo_id] = [p_late[i_halo], p_s_zero[i_halo], p_sab[i_halo]]

        values = -1 * np.ones(shape=(histories.shape[0], 3), dtype='f8')
        for i_history in range(histories.shape[0]):
            subhalo_id = int(histories.iloc[[i_history]]['subhalo_id'])
            if subhalo_id in morphology:
                values[i_history] = morphology[subhalo_id]
        histories[['p_late', 'p_s_zero', 'p_sab']] = values
    return histories


def add_averaged_sfr(histories):
    # TODO: Decide whether to implement
    sfr_filename = raw_data_dir + 'star_formation_rates.hdf5'
    return histories


extract_histories_from_file = {
    'lhalotree': lhalotree_extract_histories_from_file,
    'sublink': sublink_extract_histories_from_file,
}[config.tree_algorithm]
filenames = [raw_data_dir+name for name in os.listdir(raw_data_dir)]
assert config.n_process <= 5  # For some reason sublink will hang if n_process is large
pool = multiprocessing.Pool(config.n_process)

all_histories = []
while filenames:
    files_to_process, filenames = filenames[:config.n_process], filenames[config.n_process:]
    files_to_process = [(config, file) for file in files_to_process]
    pool_result = pool.starmap(extract_histories_from_file, files_to_process)

    log('Concatenating dataframes')
    if type(all_histories) == list:
        all_histories = pool_result.pop(0)
    while pool_result:
        all_histories = pd.concat([all_histories, pool_result.pop(0)], ignore_index=True)

all_histories = add_morphologies(all_histories, config)
all_histories = add_averaged_sfr(all_histories)

log('Saving data')
if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)
all_histories.to_pickle(generated_data_dir+'histories.pickle')
with open(generated_data_dir+'redshifts.yaml', 'w') as yaml_file:
    yaml.dump(redshifts, yaml_file)
with open(generated_data_dir+'ages.yaml', 'w') as yaml_file:
    yaml.dump(ages, yaml_file)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
