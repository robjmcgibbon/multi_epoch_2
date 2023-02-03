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

assert config.simulation in ['lgalaxies_2020']

tree_data_dir = config.get_raw_data_dir()
sam_data_dir = tree_data_dir + '../../lgalaxies/'
generated_data_dir = config.get_generated_data_dir()

ages = helpers.get_tng_ages()
redshifts = helpers.get_tng_redshifts()


def lhalotree_extract_subhalos_from_file(script_config, filename):
    log(f'Starting processing file: {filename}')

    with h5py.File(filename, 'r') as file:
        n_halos_in_tree = np.array(file['/Header/TreeNHalos'])

        subhalos = []
        for i_tree, n_halo in enumerate(n_halos_in_tree):
            tree_arr = {}
            tree = file[f'Tree{i_tree}']

            # Convert positions from kpc to Mpc
            arr_loc = np.array(tree['SubhaloPos']) / 1000
            tree_arr['x'] = arr_loc[:, 0]
            tree_arr['y'] = arr_loc[:, 1]
            tree_arr['z'] = arr_loc[:, 2]

            # Convert mass to solar units
            arr_mass_type = np.array(tree['SubhaloMassType']) * (10**10) / script_config.get_hubble_constant()
            tree_arr['dm_fof_mass'] = np.array(tree['Group_M_Crit200']) * (10**10) / script_config.get_hubble_constant()

            tree_arr['dm_sub_mass'] = arr_mass_type[:, 1]

            tree_arr['main_prog_index'] = np.array(tree['FirstProgenitor'])
            tree_arr['next_prog_index'] = np.array(tree['NextProgenitor'])
            tree_arr['snap_num'] = np.array(tree['SnapNum'])
            tree_arr['subhalo_id'] = np.array(tree['SubhaloNumber'])

            arr_central_index = np.array(tree['FirstHaloInFOFGroup'])
            tree_arr['is_central'] = np.zeros(n_halo, dtype=bool)
            for i_halo, i_central in enumerate(arr_central_index):
                tree_arr['dm_fof_mass'][i_halo] = tree_arr['dm_fof_mass'][i_central]
                tree_arr['is_central'][i_halo] = (i_halo == i_central)

            subhalos.append(pd.DataFrame(tree_arr))

    subhalos = pd.concat(subhalos)
    log(f'Finished processing file: {filename}')
    return subhalos


pool = multiprocessing.Pool(config.n_process)

files_to_process = [(config, tree_data_dir+name) for name in os.listdir(tree_data_dir)]
pool_result = pool.starmap(lhalotree_extract_subhalos_from_file, files_to_process)

log('Concatenating dataframes')
all_subhalos = pool_result.pop(0)
while pool_result:
    all_subhalos = pd.concat([all_subhalos, pool_result.pop(0)], ignore_index=True)

log('Converting dataframe into dict of numpy arrays')
arr = {key: np.array(all_subhalos[key]) for key in all_subhalos.columns}
arr['bh_mass'] = np.zeros_like(arr['dm_sub_mass'])
arr['bh_dot'] = np.zeros_like(arr['dm_sub_mass'])
arr['gas_mass'] = np.zeros_like(arr['dm_sub_mass'])
arr['sfr'] = np.zeros_like(arr['dm_sub_mass'])
arr['stellar_mass'] = np.zeros_like(arr['dm_sub_mass'])
arr['stellar_metallicity'] = np.zeros_like(arr['dm_sub_mass'])
arr['mock_g'] = np.zeros_like(arr['dm_sub_mass'])
arr['mock_k'] = np.zeros_like(arr['dm_sub_mass'])
arr['mock_r'] = np.zeros_like(arr['dm_sub_mass'])
arr['mock_u'] = np.zeros_like(arr['dm_sub_mass'])

# TODO: I could calculate the indices required in parallel
for snap in range(100):
    log(f'Extracting galaxy data for snapshot: {snap}')
    sam_filename = sam_data_dir + f'LGalaxies_{snap:03}.hdf5'

    with h5py.File(sam_filename, 'r') as sam_file:
        galaxies = sam_file['Galaxy']
        sam_subhalo_id = np.array(galaxies['SubhaloIndex_TNG-Dark'])
        sam_bh_mass = np.array(galaxies['BlackHoleMass']) * (10 ** 10) / config.get_hubble_constant()
        sam_bh_dot = np.array(galaxies['QuasarAccretionRate']) + np.array(galaxies['RadioAccretionRate'])
        # TODO: Include cold, hot gas
        sam_cold_gas_mass = np.array(galaxies['ColdGasMass']) * (10 ** 10) / config.get_hubble_constant()
        sam_hot_gas_mass = np.array(galaxies['HotGasMass']) * (10 ** 10) / config.get_hubble_constant()
        sam_gas_mass = sam_cold_gas_mass + sam_hot_gas_mass
        sam_sfr = np.array(galaxies['StarFormationRate'])
        sam_stellar_mass = np.array(galaxies['StellarMass']) * (10 ** 10) / config.get_hubble_constant()
        sam_stellar_metallicity = np.zeros_like(sam_stellar_mass)
        sam_metal_stars = np.array(galaxies['MetalsStellarMass'])[sam_stellar_mass != 0]
        sam_total_stars = np.array(galaxies['StellarMass'])[sam_stellar_mass != 0]
        sam_stellar_metallicity[sam_stellar_mass != 0] = sam_metal_stars / sam_total_stars
        sam_mag = np.array(galaxies['Mag'])
        sam_mock_g = sam_mag[:, 16]
        sam_mock_k = sam_mag[:, 9]
        sam_mock_r = sam_mag[:, 17]
        sam_mock_u = sam_mag[:, 15]

        # TODO: Don't need fof mass
        sam_fof_mass = np.array(galaxies['Central_M_Crit200']) * (10 ** 10) / config.get_hubble_constant()

    sam_dict_index_from_subhalo_id = {sub_id: i_arr for i_arr, sub_id in enumerate(sam_subhalo_id)}

    snap_mask = arr['snap_num'] == snap
    dmo_subhalo_id = arr['subhalo_id'][snap_mask]
    # For each dmo subhalo, dmo_sam_index contains the index of the subhalos properties in a sam_array
    dmo_sam_index = -1 * np.ones(dmo_subhalo_id.shape, dtype='int64')
    for i_dmo, sub_id in enumerate(dmo_subhalo_id):
        dmo_sam_index[i_dmo] = sam_dict_index_from_subhalo_id.get(sub_id, -1)
    has_match = dmo_sam_index != -1
    dmo_sam_index = dmo_sam_index[has_match]

    arr['bh_mass'][snap_mask][has_match] = sam_bh_mass[dmo_sam_index]
    arr['bh_dot'][snap_mask][has_match] = sam_bh_dot[dmo_sam_index]
    arr['gas_mass'][snap_mask][has_match] = sam_gas_mass[dmo_sam_index]
    arr['sfr'][snap_mask][has_match] = sam_sfr[dmo_sam_index]
    arr['stellar_mass'][snap_mask][has_match] = sam_stellar_mass[dmo_sam_index]
    arr['stellar_metallicity'][snap_mask][has_match] = sam_stellar_metallicity[dmo_sam_index]
    arr['mock_g'][snap_mask][has_match] = sam_mock_g[dmo_sam_index]
    arr['mock_k'][snap_mask][has_match] = sam_mock_k[dmo_sam_index]
    arr['mock_r'][snap_mask][has_match] = sam_mock_r[dmo_sam_index]
    arr['mock_u'][snap_mask][has_match] = sam_mock_u[dmo_sam_index]

    # This is a good check to make sure the galaxies have been matched properly
    dmo_fof_mass = arr['dm_fof_mass'][snap_mask][has_match]
    sam_fof_mass_test = sam_fof_mass[dmo_sam_index]
    # TODO: Why does this happen?
    nonzero_fof_mask = dmo_fof_mass != 0
    dmo_fof_mass = dmo_fof_mass[nonzero_fof_mask]
    sam_fof_mass_test = sam_fof_mass_test[nonzero_fof_mask]
    if not np.allclose(dmo_fof_mass, sam_fof_mass_test, rtol=1e-3):
        print(dmo_fof_mass)
        print(sam_fof_mass_test)
        assert np.allclose(dmo_fof_mass, sam_fof_mass_test)
    arr['dm_fof_mass'][snap_mask][has_match] = sam_fof_mass[dmo_sam_index]

log('Creating histories dataframe')
all_histories = helpers.fill_histories_dataframe(config, arr)

log('Saving data')
if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)
all_histories.to_csv(generated_data_dir+'histories.csv', index=False)
with open(generated_data_dir+'redshifts.yaml', 'w') as yaml_file:
    yaml.dump(redshifts, yaml_file)
with open(generated_data_dir+'ages.yaml', 'w') as yaml_file:
    yaml.dump(ages, yaml_file)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
