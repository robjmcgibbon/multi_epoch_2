import multiprocessing
import re
import os
import sys

import astropy.cosmology
import h5py
import numpy as np
import pandas as pd
import scipy.spatial
import yaml
import ytree

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log


def create_histories(run_dir, subfind_prop_from_rockstar_id):
    a = ytree.load(run_dir+'/tree_0_0_0.dat')
    run_name = run_dir.split('/')[-1]
    sim_name = run_dir.split('/')[-2]
    log(f'{sim_name}/{run_name}: Creating histories from rockstar merger trees')

    n_valid = 0
    factors = set()
    for root_node in a:
        if (root_node['scale_factor'] == 1) and (root_node['id'] in subfind_prop_from_rockstar_id):
            n_valid += 1
        for node in root_node['tree']:
            factors.add(node['scale_factor'])
    factors = sorted(factors)

    snapshots = list(range(len(factors)))
    redshifts = [(1/f)-1 for f in factors]
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=a.hubble_constant*100, Om0=a.omega_matter)
    ages = [cosmo.age(z).value for z in redshifts]

    redshifts = list(map(float, redshifts))
    ages = list(map(float, ages))

    snapshot_from_factor = {f: snap for snap, f in enumerate(factors)}
    ages = {snap: round(age, 2) for (snap, age) in zip(snapshots, ages)}
    redshifts = {snap: round(z, 2) for (snap, z) in zip(snapshots, redshifts)}

    input_properties = ['bh_mass', 'bh_dot', 'dm_fof_mass', 'dm_sub_mass', 'gas_mass', 'sfr', 'stellar_mass']
    output_features = ['bh_mass', 'gas_mass', 'mock_g', 'mock_k', 'mock_r', 'mock_u',
                       'sfr', 'stellar_mass', 'stellar_metallicity',
                       'central', 'lowest_snap', 'subhalo_id', 'x', 'y', 'z']

    min_snap = 0
    max_snap = 33
    snapshots = list(range(max_snap, min_snap-1, -1))
    n_input, n_output, n_snap = len(input_properties), len(output_features), len(snapshots)
    input_features = [str(snap)+prop for snap in snapshots for prop in input_properties]
    histories = np.zeros((n_valid, n_input*n_snap + n_output), dtype='float64')

    i_sub = 0
    for root_node in a:  # Looping over the trees
        if root_node['scale_factor'] != 1:
            continue
        if root_node['id'] not in subfind_prop_from_rockstar_id:
            continue

        snap_num = 33  # Stops pycharm raising a warning
        for node in root_node['prog']:  # Looping over the main progenitor branch
            snap_num = snapshot_from_factor[node['scale_factor']]
            rockstar_id = node['id']
            if rockstar_id in subfind_prop_from_rockstar_id:
                bh_mass = subfind_prop_from_rockstar_id[rockstar_id]['bh_mass']
                bh_dot = subfind_prop_from_rockstar_id[rockstar_id]['bh_dot']
                dm_fof_mass = subfind_prop_from_rockstar_id[rockstar_id]['dm_fof_mass']
                dm_sub_mass = subfind_prop_from_rockstar_id[rockstar_id]['dm_sub_mass']
                gas_mass = subfind_prop_from_rockstar_id[rockstar_id]['gas_mass']
                sfr = subfind_prop_from_rockstar_id[rockstar_id]['sfr']
                stellar_mass = subfind_prop_from_rockstar_id[rockstar_id]['stellar_mass']

                data = [bh_mass, bh_dot, dm_fof_mass, dm_sub_mass, gas_mass, sfr, stellar_mass]
            else:
                data = [0, 0, 0, 0, 0, 0, 0]

            i_start = (max_snap - snap_num) * n_input
            histories[i_sub, i_start:i_start+n_input] = data

        bh_mass = subfind_prop_from_rockstar_id[root_node['id']]['bh_mass']
        gas_mass = subfind_prop_from_rockstar_id[root_node['id']]['gas_mass']
        mock_g = subfind_prop_from_rockstar_id[root_node['id']]['mock_g']
        mock_k = subfind_prop_from_rockstar_id[root_node['id']]['mock_k']
        mock_r = subfind_prop_from_rockstar_id[root_node['id']]['mock_r']
        mock_u = subfind_prop_from_rockstar_id[root_node['id']]['mock_u']
        sfr = subfind_prop_from_rockstar_id[root_node['id']]['sfr']
        stellar_mass = subfind_prop_from_rockstar_id[root_node['id']]['stellar_mass']
        stellar_metallicity = subfind_prop_from_rockstar_id[root_node['id']]['stellar_metallicity']

        # lowest_snap gives the highest z rockstar halo, does not mean there is a subfind match
        lowest_snap = snap_num
        x = float(root_node['x'].to('Mpc'))
        y = float(root_node['y'].to('Mpc'))
        z = float(root_node['z'].to('Mpc'))
        subhalo_id = root_node['id']
        central = subfind_prop_from_rockstar_id[root_node['id']]['central']

        data = [bh_mass, gas_mass, mock_g, mock_k, mock_r, mock_u,
                sfr, stellar_mass, stellar_metallicity,
                central, lowest_snap, subhalo_id, x, y, z]
        histories[i_sub, n_input*n_snap:] = data
        i_sub += 1

    histories = pd.DataFrame(histories, columns=input_features+output_features)

    # Filling missing data using snapshots above and below
    # Notice that the snapshots is now in ascending order
    has_match = np.zeros((histories.shape[0], len(snapshots)), dtype=bool)
    for i_snap, snap in enumerate(snapshots[::-1]):
        has_match[:, i_snap] = (histories[str(snap)+'dm_sub_mass'] != 0)
    for i_snap in range(1, len(snapshots)-1):
        below_snap, snap, above_snap = i_snap-1, i_snap, i_snap+1
        # has_match_above contains halos not matched at current snapshot, but are matched at next snapshot
        has_match_above = np.logical_not(has_match[:, i_snap]) & has_match[:, i_snap+1]
        # has_match_below contains halos not matched at current or next snapshot, but are matched at previous snapshot
        has_match_below = np.logical_not(has_match[:, i_snap]) & np.logical_not(has_match[:, i_snap+1])
        has_match_below = has_match_below & has_match[:, i_snap-1]

        for prop in input_properties:
            arr = np.array(histories[str(snap)+prop])
            arr[has_match_above] = np.array(histories[str(above_snap)+prop])[has_match_above]
            arr[has_match_below] = np.array(histories[str(below_snap)+prop])[has_match_below]
            histories[str(snap)+prop] = arr

    generated_data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim_name}/{run_name}/'
    if not os.path.exists(generated_data_dir):
        os.makedirs(generated_data_dir)
    histories.to_pickle(generated_data_dir+'histories.pickle')
    with open(generated_data_dir+'ages.yaml', 'w') as yaml_file:
        yaml.dump(ages, yaml_file)
    with open(generated_data_dir+'redshifts.yaml', 'w') as yaml_file:
        yaml.dump(redshifts, yaml_file)

    return 0


def match_subfind_to_rockstar(run_dir):
    a = ytree.load(run_dir+'/tree_0_0_0.dat')
    run_name = run_dir.split('/')[-1]
    sim_name = run_dir.split('/')[-2]
    log(f'{sim_name}/{run_name}: Extracting subfind data and matching to rockstar')

    with open(run_dir+'/tree_0_0_0.dat', 'r') as file:
        names = file.readline().rstrip()[1:]  # Remove leading #
        names = [re.sub('\(\d*\)', '', n) for n in names.split(' ')]  # Reformat names, e.g. id(1) -> id
    rockstar_data = pd.read_csv(run_dir+'/tree_0_0_0.dat', names=names, comment='#', delim_whitespace=True)
    rockstar_data = rockstar_data.dropna()  # One line gives the number of trees, this removes it

    rockstar_scale_factors = np.array(rockstar_data['scale'])
    rockstar_ids = np.array(rockstar_data['id'], dtype=int)
    x = np.array(rockstar_data['x']) / a.hubble_constant
    y = np.array(rockstar_data['y']) / a.hubble_constant
    z = np.array(rockstar_data['z']) / a.hubble_constant
    rockstar_pos = np.array([x, y, z]).T
    rockstar_mass = np.array(rockstar_data['Mvir']) / a.hubble_constant
    rockstar_stellar = np.array(rockstar_data['SM']) / a.hubble_constant
    rockstar_gas = np.array(rockstar_data['Gas']) / a.hubble_constant

    box_length = 25 / a.hubble_constant
    scale_factors = sorted(set(rockstar_scale_factors))
    snapshots = list(range(len(scale_factors)))
    snapshot_from_factor = {f: snap for snap, f in enumerate(scale_factors)}
    rockstar_snapshots = np.array([snapshot_from_factor[f] for f in rockstar_scale_factors])

    n_subfind_per_snap = {}
    subfind_prop_from_rockstar_id = {}
    for snap in snapshots:
        with h5py.File(run_dir+f'/fof_subhalo_tab_{snap:03d}.hdf5', 'r') as file:

            subfind_mass_type = np.array(file['/Subhalo/SubhaloMassType']) * (10 ** 10) / a.hubble_constant
            subfind_pos = np.array(file['/Subhalo/SubhaloPos']) / (1000 * a.hubble_constant)
            subfind_half_mass_rad = np.array(file['/Subhalo/SubhaloHalfmassRad']) / (1000 * a.hubble_constant)

            subfind_bh_mass = np.array(file['/Subhalo/SubhaloBHMass']) * (10 ** 10) / a.hubble_constant
            subfind_bh_dot = np.array(file['/Subhalo/SubhaloBHMdot']) * 10.22
            subfind_dm_sub_mass = subfind_mass_type[:, 1]
            subfind_gas_mass = subfind_mass_type[:, 0]
            subfind_mock_g = np.array(file['/Subhalo/SubhaloStellarPhotometrics'])[:, 4]
            subfind_mock_k = np.array(file['/Subhalo/SubhaloStellarPhotometrics'])[:, 3]
            subfind_mock_r = np.array(file['/Subhalo/SubhaloStellarPhotometrics'])[:, 5]
            subfind_mock_u = np.array(file['/Subhalo/SubhaloStellarPhotometrics'])[:, 0]
            subfind_sfr = np.array(file['/Subhalo/SubhaloSFR'])
            subfind_stellar_mass = subfind_mass_type[:, 4]
            subfind_stellar_metallicity = np.array(file['/Subhalo/SubhaloStarMetallicity'])

            subfind_i_fof = np.array(file['/Subhalo/SubhaloGrNr'])
            group_dm_fof_mass = np.array(file['/Group/Group_M_Crit200']) * (10 ** 10) / a.hubble_constant
            subfind_dm_fof_mass = group_dm_fof_mass[subfind_i_fof]
            group_first_sub = np.array(file['/Group/GroupFirstSub'])
            subfind_central = group_first_sub[subfind_i_fof] == np.arange(subfind_i_fof.shape[0])

            subfind_mass = np.sum(subfind_mass_type, axis=1)
            mass_cuts = [10**8, 10**9, 10**10, 10**11, 10**12, 10**13]
            for mass_cut in mass_cuts:
                n_subfind_per_snap[f'{snap}_{mass_cut:.2g}'] = np.sum(subfind_mass > mass_cut)

        snap_rockstar_mass = rockstar_mass[rockstar_snapshots == snap]
        snap_rockstar_ids = rockstar_ids[rockstar_snapshots == snap]
        snap_rockstar_pos = rockstar_pos[rockstar_snapshots == snap]
        # boxsize multiplication factor is needed because there is a halo with a pos exactly equal to box length
        rockstar_kdtree = scipy.spatial.KDTree(snap_rockstar_pos, boxsize=box_length*1.00001)
        for i_sub, pos in enumerate(subfind_pos):
            # TODO: Set r frac
            r = subfind_half_mass_rad[i_sub] * 3
            sub_m = subfind_dm_sub_mass[i_sub]
            close_rockstar_halos = rockstar_kdtree.query_ball_point(pos, r)
            close_rockstar_halos = [i_rock for i_rock in close_rockstar_halos
                                    if ((sub_m/3) < snap_rockstar_mass[i_rock])
                                    and ((sub_m*3) > snap_rockstar_mass[i_rock])]
            if not close_rockstar_halos:
                continue
            close_rockstar_halos = sorted(close_rockstar_halos,
                                          key=lambda i_rock: np.linalg.norm(snap_rockstar_pos[i_rock] - pos))
            subfind_prop = {
                'bh_mass': subfind_bh_mass[i_sub],
                'bh_dot': subfind_bh_dot[i_sub],
                'dm_fof_mass': subfind_dm_fof_mass[i_sub],
                'dm_sub_mass': subfind_dm_sub_mass[i_sub],
                'gas_mass': subfind_gas_mass[i_sub],
                'mock_g': subfind_mock_g[i_sub],
                'mock_k': subfind_mock_k[i_sub],
                'mock_r': subfind_mock_r[i_sub],
                'mock_u': subfind_mock_u[i_sub],
                'sfr': subfind_sfr[i_sub],
                'stellar_mass': subfind_stellar_mass[i_sub],
                'stellar_metallicity': subfind_stellar_metallicity[i_sub],
                'central': subfind_central[i_sub],
            }
            if snap_rockstar_ids[close_rockstar_halos[0]] in subfind_prop_from_rockstar_id:
                # TODO: Pick closest
                # Checks if a rockstar halo is already matched. If it is, pick the closest subhalo.
                # I need to save pos to subfind_prop for this to work
                # matched_pos = subfind_prop_from_rockstar_id[snap_rockstar_ids[close_rockstar_halos[0]]]['pos']
                # matched_dist = np.linalg.norm(snap_rockstar_pos[close_rockstar_halos[0]] - matched_pos)
                # dist = np.linalg.norm(snap_rockstar_pos[close_rockstar_halos[0]] - pos)
                # if dist < matched_dist:
                #     subfind_prop_from_rockstar_id[snap_rockstar_ids[close_rockstar_halos[0]]] = subfind_prop

                # Combine subhalos when there are multiple matches
                matched_prop = subfind_prop_from_rockstar_id[snap_rockstar_ids[close_rockstar_halos[0]]]
                combined_prop = {'bh_mass': matched_prop['bh_mass'] + subfind_prop['bh_mass']}
                if combined_prop['bh_mass']:
                    combined_prop['bh_dot'] = matched_prop['bh_mass'] * matched_prop['bh_dot']
                    combined_prop['bh_dot'] += subfind_prop['bh_mass'] * subfind_prop['bh_dot']
                    combined_prop['bh_dot'] /= combined_prop['bh_mass']
                combined_prop['dm_fof_mass'] = subfind_prop['dm_fof_mass']
                combined_prop['dm_sub_mass'] = matched_prop['dm_sub_mass'] + subfind_prop['dm_sub_mass']
                combined_prop['gas_mass'] = matched_prop['gas_mass'] + subfind_prop['gas_mass']
                combined_prop['stellar_mass'] = matched_prop['stellar_mass'] + subfind_prop['stellar_mass']
                combined_prop['central'] = matched_prop['central'] or subfind_prop['central']
                if combined_prop['stellar_mass']:
                    for prop in ['mock_g', 'mock_k', 'mock_r', 'mock_u', 'sfr', 'stellar_metallicity']:
                        combined_prop[prop] = matched_prop['stellar_mass'] * matched_prop[prop]
                        combined_prop[prop] += subfind_prop['stellar_mass'] * subfind_prop[prop]
                        combined_prop[prop] /= combined_prop['stellar_mass']
            # TODO: subfind_prop_from_rockstar_id[snap_rockstar_ids[close_rockstar_halos[0]]] = combined_prop
            else:
                subfind_prop_from_rockstar_id[snap_rockstar_ids[close_rockstar_halos[0]]] = subfind_prop

    data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim_name}/{run_name}/matching/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    matching_stats = {}
    for plot_snap in [4, 10, 18, 33]:  # Corresponds to z=3,2,1,0
        r_matched_mass, s_matched_mass, r_unmatched_mass = [], [], []
        r_matched_gas, s_matched_gas, r_unmatched_gas = [], [], []
        r_matched_stellar, s_matched_stellar, r_unmatched_stellar = [], [], []
        for i_rock, rock_snap in enumerate(rockstar_snapshots):
            if rock_snap != plot_snap:
                continue
            try:
                subfind_prop = subfind_prop_from_rockstar_id[rockstar_ids[i_rock]]
                sub_mass = sum([subfind_prop['bh_mass'],
                                subfind_prop['gas_mass'],
                                subfind_prop['dm_sub_mass'],
                                subfind_prop['stellar_mass']])
                r_matched_mass.append(float(rockstar_mass[i_rock]))
                s_matched_mass.append(float(sub_mass))
                r_matched_gas.append(float(rockstar_gas[i_rock]))
                s_matched_gas.append(float(subfind_prop['gas_mass']))
                r_matched_stellar.append(float(rockstar_stellar[i_rock]))
                s_matched_stellar.append(float(subfind_prop['stellar_mass']))
            except KeyError:
                r_unmatched_mass.append(float(rockstar_mass[i_rock]))
                r_unmatched_gas.append(float(rockstar_gas[i_rock]))
                r_unmatched_stellar.append(float(rockstar_stellar[i_rock]))
        np.save(f'{data_dir}{plot_snap}_r_matched_mass', r_matched_mass)
        np.save(f'{data_dir}{plot_snap}_s_matched_mass', s_matched_mass)
        np.save(f'{data_dir}{plot_snap}_r_unmatched_mass', r_unmatched_mass)
        np.save(f'{data_dir}{plot_snap}_r_matched_gas', r_matched_gas)
        np.save(f'{data_dir}{plot_snap}_s_matched_gas', s_matched_gas)
        np.save(f'{data_dir}{plot_snap}_r_unmatched_gas', r_unmatched_gas)
        np.save(f'{data_dir}{plot_snap}_r_matched_stellar', r_matched_stellar)
        np.save(f'{data_dir}{plot_snap}_s_matched_stellar', s_matched_stellar)
        np.save(f'{data_dir}{plot_snap}_r_unmatched_stellar', r_unmatched_stellar)

        for mass_cut in mass_cuts:
            n_matched = 0
            snap_rockstar_mass = rockstar_mass[rockstar_snapshots == plot_snap]
            snap_rockstar_ids = rockstar_ids[rockstar_snapshots == plot_snap]
            for r_mass, r_id in zip(snap_rockstar_mass, snap_rockstar_ids):
                if r_mass > mass_cut and r_id in subfind_prop_from_rockstar_id:
                    n_matched += 1
            n_rockstar = np.sum(rockstar_mass[rockstar_snapshots == plot_snap] > mass_cut)
            n_subfind = n_subfind_per_snap[f'{plot_snap}_{mass_cut:.2g}']
            matching_stats[f'{plot_snap}_n_matched_{mass_cut:.2g}'] = int(n_matched)
            matching_stats[f'{plot_snap}_n_rockstar_{mass_cut:.2g}'] = int(n_rockstar)
            matching_stats[f'{plot_snap}_n_subfind_{mass_cut:.2g}'] = int(n_subfind)

    with open(data_dir+'stats.yaml', 'w') as yaml_file:
        yaml.dump(matching_stats, yaml_file)

    return subfind_prop_from_rockstar_id


def extract_camels(run_dir):
    subfind_prop_from_rockstar_id = match_subfind_to_rockstar(run_dir)
    create_histories(run_dir, subfind_prop_from_rockstar_id)
    return 0


pool = multiprocessing.Pool(5)
for sim in ['IllustrisTNG', 'SIMBA']:
    raw_data_dir = f'{helpers.Config.get_base_dir()}downloaded/camels/{sim}/'
    run_names = sorted(os.listdir(raw_data_dir))
    run_dirs = [raw_data_dir+run_name for run_name in run_names]
    results = pool.map(extract_camels, run_dirs)

log('Job finished')
