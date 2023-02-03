import datetime
import os
import sys

import numpy as np
import pandas as pd

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')
generated_data_dir = config.get_generated_data_dir()
histories = pd.read_pickle(generated_data_dir+'histories.pickle')

# This script is for comparing TNG100-2 with camels (they have similar resolution) to see how accurate matching is
# Snapshots correspond to z=3, 2, 1, 0
# You need to reextract TNG100-2 with no mass limit before running this
for mass_cut in [10**9, 10**10]:
    log(f'Mass cut: {mass_cut:.2g}')
    mass_cut_histories = histories[histories['99dm_sub_mass'] > mass_cut]
    snaps = [25, 33, 50, 99]
    for snap in snaps:
        frac_has_merger_tree_halo = np.sum(mass_cut_histories['lowest_snap'] <= snap)/mass_cut_histories.shape[0]
        frac_has_stellar_mass = np.sum(mass_cut_histories[str(snap)+'stellar_mass'] != 0)/mass_cut_histories.shape[0]
        log(f'Fraction with merger tree back to snapshot {snap}: {frac_has_merger_tree_halo:.2g}')
        log(f'Fraction with stellar mass at snapshot {snap}: {frac_has_stellar_mass:.2g}')

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
