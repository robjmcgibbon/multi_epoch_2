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
main_config = helpers.Config('config.yaml')
secondary_configs = main_config.load_secondary_configs('config.yaml')

fig, ax = plt.subplots(1)
bin_width = .2
bins = np.arange(7.9, 12.11, bin_width)
mids = (bins[:-1] + bins[1:]) / 2

for i_config, config in enumerate([main_config] + secondary_configs):
    log(f'Plotting for {config.name}')

    snapshots = config.get_standard_spacing()
    data = config.load_data(snapshots)

    data = data[data['stellar_mass'] != 0]
    stellar_mass = np.array(data['stellar_mass'])
    stellar_mass = np.log10(stellar_mass)

    x, y, z = np.array(data['x']), np.array(data['y']), np.array(data['z'])
    midpoint = config.box_length / 2
    full_volume = config.box_length ** 3

    vals = []
    for x_mask in [x < midpoint, x > midpoint]:
        for y_mask in [y < midpoint, y > midpoint]:
            for z_mask in [z < midpoint, z > midpoint]:
                # mask = np.logical_not(x_mask & y_mask & z_mask)
                mask = x_mask & y_mask & z_mask
                volume = full_volume * 1 / 8
                weights = np.full_like(stellar_mass[mask], 1 / (volume * bin_width))
                n, _ = np.histogram(stellar_mass[mask], bins=bins, weights=weights)
                vals.append(n)
    vals = np.array(vals)
    spread = np.std(vals, axis=0)

    full_weights = np.full_like(stellar_mass, 1 / (full_volume * bin_width))
    n, _ = np.histogram(stellar_mass, bins=bins, weights=full_weights)
    mask = n != 0
    p = ax.plot(mids[mask], n[mask], '--', label=config.name)

    ax.fill_between(mids[mask], n[mask]-spread[mask], n[mask]+spread[mask],
                    color=p[0].get_color(),
                    # label=config.name,
                    alpha=0.2, edgecolor=None)

# mids = np.load('/home/rmcg/simba_mids.npy')
# n = np.load('/home/rmcg/simba_n.npy')
# spread = np.load('/home/rmcg/simba_spread.npy')
# p = ax.plot(mids, n, '--', label='Simba')
# ax.fill_between(mids, n-spread, n+spread,
                # color=p[0].get_color(), alpha=0.2, edgecolor=None)

ax.set_xlabel('$\log_{10}\, M_{*}$  $[M_\odot]$', fontsize=13)
ax.set_xlim(8, 12)
ylabel = '$\phi = \mathrm{dn}/\mathrm{dlog}_{10} \mathrm{M}_{*} \, [\mathrm{Mpc}^{-3}]$'
ax.set_ylabel(ylabel, fontsize=13)
ax.set_ylim(2*10**-5.5, 10**-0.8)
ax.set_yscale('log')
ax.legend(fontsize=13)
main_config.plot_show_save('btml_stellar_mass_function', fig, force_save=True)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
