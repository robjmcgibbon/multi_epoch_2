import datetime
import os
import sys

import astropy.cosmology
import numpy as np
import pandas as pd
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log
from virgodb import VirgoDB

log(f'Running {__file__}')
config = helpers.Config('config.yaml')

assert config.simulation.split('_')[0] == 'eagle'
generated_data_dir = config.get_generated_data_dir()

with open('virgo_credentials.yaml', 'r') as file:
    virgo_credentials = yaml.safe_load(file)

username = virgo_credentials['username']
password = virgo_credentials['password']
url = 'http://virgodb.dur.ac.uk:8080/Eagle/'
vdb = VirgoDB(username, password, url)

subhalo_table = {
    100: {
        'eagle_ref': 'RefL0100N1504_Subhalo'
    },
    50: {
        'eagle_ref': 'RefL0050N0752_Subhalo',
        'eagle_AGNdT': 'AGNdT9L0050N0752_Subhalo',
        'eagle_C15': 'Physics_vars..C15AGNdT8L0050N0752_Subhalo',
        'eagle_FBconst': 'Physics_vars..FBconstL0050N0752_Subhalo',
        'eagle_FBsigma': 'Physics_vars..FBsigmaL0050N0752_Subhalo',
        'eagle_FBZ': 'Physics_vars..FBZL0050N0752_Subhalo',
        'eagle_ViscHi': 'Physics_vars..ViscHiL0050N0752_Subhalo',
        'eagle_ViscLo': 'Physics_vars..ViscLoL0050N0752_Subhalo',
        'eagle_NoAGN': 'Physics_vars..NoAGNL0050N0752_Subhalo',
        'eagle_HiML': 'Physics_vars..HiML0050N0752_Subhalo',
        'eagle_LoML': 'Physics_vars..LoML0050N0752_Subhalo'
    }
}[config.box_size][config.simulation]
fof_table = subhalo_table.replace('Subhalo', 'FOF')

log('Querying snapshot information')
query = f'SELECT DISTINCT snapnum from {subhalo_table}'
snapshots = vdb.execute_query(query)
snapshots = np.sort([x[0] for x in snapshots])
log('Querying redshift information')
query = f'SELECT DISTINCT redshift from {subhalo_table}'
redshifts = vdb.execute_query(query)
redshifts = np.sort([x[0] for x in redshifts])[::-1]
log('Calculating ages from redshifts')
eagle_cosmology = astropy.cosmology.FlatLambdaCDM(H0=67.77, Om0=0.307)
ages = [eagle_cosmology.age(z).value for z in redshifts]

snapshots = list(map(int, snapshots))
redshifts = list(map(float, redshifts))
ages = list(map(float, ages))

ages = {snap: round(age, 2) for (snap, age) in zip(snapshots, ages)}
redshifts = {snap: round(z, 2) for (snap, z) in zip(snapshots, redshifts)}

result = []
for low_lim in np.arange(0, 1, 0.2):
    # If you want a random sample use SELECT top N along with ORDER BY NEWID()
    # This version of SQL is case-insensitive
    query = f'''
    SELECT
      sub.BlackholeMass as bh_mass,
      sub.BlackholeMassAccretionRate as bh_dot,
      sub.Snapnum as snap_num,
      sub.CentreOfMass_x as x,
      sub.CentreOfMass_y as y,
      sub.CentreOfMass_z as z,
      sub.MassType_DM as dm_sub_mass,
      sub.MassType_Gas as gas_mass,
      sub.MassType_Star as stellar_mass,
      sub.StarFormationRate as sfr,
      sub.Stars_Metallicity as stellar_metallicity,
      sub.SubGroupNumber as subgroup_num,
      sub.DescendantID as descendant_id,
      sub.GalaxyID as subhalo_id,
      fof.Group_M_Crit200 as dm_fof_mass
    FROM
      {subhalo_table} as sub,
      {fof_table} as fof
    WHERE
      sub.Spurious = 0
      and fof.groupid = sub.groupid
      and sub.RandomNumber >= {low_lim}
      and sub.RandomNumber < {low_lim + 0.2}
    '''

    log('Downloading data from VirgoDB')
    q_result = vdb.execute_query(query)
    q_result = pd.DataFrame(q_result)
    if type(result) == list:
        result = q_result
    else:
        result = pd.concat([result, q_result], ignore_index=True)

n_halo = result.shape[0]
arr = pd.DataFrame({
    'x': np.array(result['x']),
    'y': np.array(result['y']),
    'z': np.array(result['z']),
    'bh_mass': np.array(result['bh_mass']),
    'bh_dot': np.array(result['bh_dot']),
    'dm_fof_mass': np.array(result['dm_fof_mass']),
    'dm_sub_mass': np.array(result['dm_sub_mass']),
    'gas_mass': np.array(result['gas_mass']),
    'sfr': np.array(result['sfr']),
    'stellar_mass': np.array(result['stellar_mass']),
    'stellar_metallicity': np.array(result['stellar_metallicity']),
    'snap_num': np.array(result['snap_num']),
    'subhalo_id': np.array(result['subhalo_id']),
})

arr_subhalo_id = np.array(result['subhalo_id'])
dict_index_from_id = {sub_id: i_arr for i_arr, sub_id in enumerate(arr_subhalo_id)}

log(f'Creating merger tree arrays')
arr_dm_sub_mass = np.array(result['dm_sub_mass'])
arr_descendant_id = np.array(result['descendant_id'])
all_progenitors = {}
for index, descendant_id in enumerate(arr_descendant_id):
    if descendant_id not in dict_index_from_id:
        continue  # Spurious halos can still be set as descendants
    if dict_index_from_id[descendant_id] == index:
        continue  # For eagle_ref galaxies can refer to themselves as descendants
    try:
        prog_indices = all_progenitors[descendant_id]
        # Make sure that the first item of prog_indices is the most massive progenitor
        # There is no order to the other items of prog_indices
        if arr_dm_sub_mass[index] > arr_dm_sub_mass[prog_indices[0]]:
            prog_indices = [index] + prog_indices
        else:
            prog_indices = prog_indices + [index]
    except KeyError:
        prog_indices = [index]
    all_progenitors[descendant_id] = prog_indices

arr_main_prog_index = -1 * np.ones(n_halo, dtype='int64')
arr_next_prog_index = -1 * np.ones(n_halo, dtype='int64')
for descendant_id, prog_indices in all_progenitors.items():
    descendant_index = dict_index_from_id[descendant_id]
    arr_main_prog_index[descendant_index] = prog_indices[0]
    for prog_1, prog_2 in zip(prog_indices, prog_indices[1:] + [-1]):
        arr_next_prog_index[prog_1] = prog_2

arr['is_central'] = np.array(result['subgroup_num'] == 0)
arr['main_prog_index'] = arr_main_prog_index
arr['next_prog_index'] = arr_next_prog_index

# Magnitude values are set in extract_eagle
arr['mock_g'] = np.ones(n_halo, dtype='f8')
arr['mock_k'] = np.ones(n_halo, dtype='f8')
arr['mock_r'] = np.ones(n_halo, dtype='f8')
arr['mock_u'] = np.ones(n_halo, dtype='f8')

# TODO: Add in size and morphology information

# Have to save data for two reasons
#   - Downloading must be done on cuillin login node, but I don't want to do processing on this node
#   - I don't want to redownload the same data when I'm predicting redshift properties for a different snapshot
log('Saving data')
data_dir = generated_data_dir + '../download_eagle/'  # Data extracted here is not for a specifc redshift
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
arr.to_pickle(data_dir+'arr.pickle')
with open(data_dir+'redshifts.yaml', 'w') as yaml_file:
    yaml.dump(redshifts, yaml_file)
with open(data_dir+'ages.yaml', 'w') as yaml_file:
    yaml.dump(ages, yaml_file)

if config.simulation in ['eagle_ref', 'eagle_AGNdT']:
    log('Downloading magnitude data')
    mag_table = subhalo_table.replace('Subhalo', 'Magnitudes')
    query = f'''
SELECT
  g_nodust as mock_g,
  k_nodust as mock_k,
  r_nodust as mock_r,
  u_nodust as mock_u,
  galaxyid as subhalo_id
FROM
  {mag_table}
'''
    mag_data = vdb.execute_query(query)
    mag_data = pd.DataFrame(mag_data)
    log('Saving magnitude data')
    mag_data.to_pickle(data_dir+'mag_data.pickle')

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
