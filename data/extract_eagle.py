import datetime
import os
import sys

import numpy as np
import pandas as pd
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')

assert config.simulation.split('_')[0] == 'eagle'
generated_data_dir = config.get_generated_data_dir()
data_dir = generated_data_dir + '../download_eagle/'  # Data extracted to this dir is not for a specific redshift

log('Loading data')
arr = pd.read_pickle(data_dir+'arr.pickle')
with open(data_dir+'redshifts.yaml', 'r') as yaml_file:
    redshifts = yaml.safe_load(yaml_file)
with open(data_dir+'ages.yaml', 'r') as yaml_file:
    ages = yaml.safe_load(yaml_file)

log('Converting dataframe into dict of numpy arrays')
arr = {col: np.array(arr[col]) for col in arr.keys()}

log('Creating histories dataframe')
histories = helpers.fill_histories_dataframe(config, arr)

if os.path.exists(data_dir+'mag_data.pickle'):
    log('Adding magnitude data')
    mag_data = pd.read_pickle(data_dir+'mag_data.pickle')
    arr_subhalo_id = np.array(mag_data['subhalo_id'])
    mock_g = np.array(mag_data['mock_g'])
    mock_k = np.array(mag_data['mock_k'])
    mock_r = np.array(mag_data['mock_r'])
    mock_u = np.array(mag_data['mock_u'])
    magnitudes = {}
    for i_halo, subhalo_id in enumerate(arr_subhalo_id):
        magnitudes[subhalo_id] = [mock_g[i_halo], mock_k[i_halo], mock_r[i_halo], mock_u[i_halo]]

    values = np.ones(shape=(histories.shape[0], 4), dtype='f8')
    for i_history in range(histories.shape[0]):
        subhalo_id = int(histories.iloc[[i_history]]['subhalo_id'])
        if subhalo_id in magnitudes:
            values[i_history] = magnitudes[subhalo_id]
    histories[['mock_g', 'mock_k', 'mock_r', 'mock_u']] = values

log('Saving data')
if not os.path.exists(generated_data_dir):
    os.makedirs(generated_data_dir)
histories.to_pickle(generated_data_dir+'histories.pickle')
with open(generated_data_dir+'redshifts.yaml', 'w') as yaml_file:
    yaml.dump(redshifts, yaml_file)
with open(generated_data_dir+'ages.yaml', 'w') as yaml_file:
    yaml.dump(ages, yaml_file)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
