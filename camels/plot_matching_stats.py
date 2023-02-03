import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log


def get_old_1p_name(name):
    _, param_i, param_value = name.split('_')
    j = ['n5', 'n4', 'n3', 'n2', 'n2', '0', '1', '2', '3', '4', '5'].index(param_value)
    return f'1P_{(int(param_i)-1)*11 + j}'


base_dir = helpers.Config.get_base_dir()
for sim in ['IllustrisTNG', 'SIMBA']:
    run_names = sorted(os.listdir(f'{base_dir}generated/baryon_tree_ml/camels/{sim}/'))

    with open(f'{base_dir}downloaded/camels/{sim}_params.txt', 'r') as txt_file:
        lines = [line.strip() for line in txt_file]
        file_run_names = [line.split(' ')[0] for line in lines]
        file_params = np.array([list(map(float, line.split(' ')[1:-1])) for line in lines])

    param_names = ['$\Omega_m$', '$\sigma_8$', '$A_{SN1}$', '$A_{AGN1}$', '$A_{SN2}$', '$A_{AGN2}$']
    params = np.zeros((len(run_names), 6), dtype=float)  # Ω_m, σ_8, A_sn1, A_agn1, A_sn2, A_agn2
    matching_stats = []
    for i_run, run_name in enumerate(run_names):
        log(f'Loading data for {sim}/{run_name}')
        data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/{run_name}/matching/'
        with open(data_dir+'stats.yaml', 'r') as yaml_file:
            matching_stats.append(yaml.safe_load(yaml_file))
        params[i_run] = file_params[file_run_names.index(run_name)]

    # Extract the mass_cuts and snapshots which are available
    mass_cuts, plot_snaps = set(), set()
    for key in matching_stats[0].keys():
        mass_match = re.search('n_matched_(.*)$', key)
        if mass_match:
            mass_cuts.add(int(float(mass_match.group(1))))
        snap_match = re.search('^(\d*)_n_matched', key)
        if snap_match:
            plot_snaps.add(int(snap_match.group(1)))

    log(f'{sim}: Snapshots to plot for: {plot_snaps}')
    for plot_snap in plot_snaps:
        log(f'{sim}: Plotting for snapshot {plot_snap}')

        # TODO: This loop has a memory leak. See https://github.com/matplotlib/matplotlib/issues/20490
        for mass_cut in mass_cuts:
            n_matched = np.zeros(len(run_names), dtype=int)
            n_rockstar = np.zeros(len(run_names), dtype=int)
            n_subfind = np.zeros(len(run_names), dtype=int)
            for i_run, run_matching_stats in enumerate(matching_stats):
                n_matched[i_run] = run_matching_stats[f'{plot_snap}_n_matched_{mass_cut:.2g}']
                n_rockstar[i_run] = run_matching_stats[f'{plot_snap}_n_rockstar_{mass_cut:.2g}']
                n_subfind[i_run] = run_matching_stats[f'{plot_snap}_n_subfind_{mass_cut:.2g}']

            plot_dir = f'/home/rmcg/camels_matching/{sim}/snapshot_{plot_snap}/n_match/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            _, ax = plt.subplots(1, dpi=400)
            sorted_args = np.argsort(n_rockstar)
            ax.plot(np.arange(len(run_names)), n_rockstar[sorted_args], '-', label='N rockstar')
            ax.plot(np.arange(len(run_names)), n_subfind[sorted_args], '-', label='N subfind')
            ax.plot(np.arange(len(run_names)), n_matched[sorted_args], '-', label='N matched')
            ax.legend()
            ax.set_xlabel('Simulation (sorted by N rockstar)')
            ax.set_ylabel('N')
            ax.set_title(f'{sim} data, Mass cut: {mass_cut:.2g}')
            plt.savefig(f'{plot_dir}mass_cut_{mass_cut:.2g}.png')
            plt.close()

            for i, param_name in enumerate(param_names):
                nice_param_name = param_name.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
                plot_dir = f'/home/rmcg/camels_matching/{sim}/snapshot_{plot_snap}/n_match_v_{nice_param_name}/'
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                _, ax = plt.subplots(1, dpi=400)
                sorted_args = np.argsort(params[:, i])
                ax.plot(params[:, i][sorted_args], n_rockstar[sorted_args], '-', label='N rockstar')
                ax.plot(params[:, i][sorted_args], n_subfind[sorted_args], '-', label='N subfind')
                ax.plot(params[:, i][sorted_args], n_matched[sorted_args], '-', label='N matched')
                ax.set_xlabel(param_name)
                ax.set_ylabel('N')
                ax.set_title(f'{sim} data, Mass cut: {mass_cut:.2g}')
                ax.legend()

                plt.savefig(f'{plot_dir}mass_cut_{mass_cut:.2g}.png')
                plt.close()

        # There's no point in plotting these for all the simulations
        for run_matching_stats, run_name in zip(matching_stats[:10], run_names[:10]):
            plot_dir = f'/home/rmcg/camels_matching/{sim}/snapshot_{plot_snap}/mass_comparisons/{run_name}/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/{run_name}/matching/'
            r_matched_mass = np.load(f'{data_dir}{plot_snap}_r_matched_mass.npy')
            s_matched_mass = np.load(f'{data_dir}{plot_snap}_s_matched_mass.npy')
            r_unmatched_mass = np.load(f'{data_dir}{plot_snap}_r_unmatched_mass.npy')
            r_matched_gas = np.load(f'{data_dir}{plot_snap}_r_matched_gas.npy')
            s_matched_gas = np.load(f'{data_dir}{plot_snap}_s_matched_gas.npy')
            r_unmatched_gas = np.load(f'{data_dir}{plot_snap}_r_unmatched_gas.npy')
            r_matched_stellar = np.load(f'{data_dir}{plot_snap}_r_matched_stellar.npy')
            s_matched_stellar = np.load(f'{data_dir}{plot_snap}_s_matched_stellar.npy')
            r_unmatched_stellar = np.load(f'{data_dir}{plot_snap}_r_unmatched_stellar.npy')

            # TODO: Raise the dpi if I'm going to use these plots for something
            _, ax = plt.subplots(1)
            ax.plot(r_matched_mass, s_matched_mass, '.')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Subfind mass [$M_\odot$]', fontsize=13)
            ax.set_ylabel('Rockstar mass [$M_\odot$]', fontsize=13)
            ax.plot([10**9, 10**14], [10**9, 10**14], 'k--')
            frac = r_matched_mass.shape[0] / (r_matched_mass.shape[0]+r_unmatched_mass.shape[0])
            log(f'Fraction of snapshot {plot_snap} rockstar halos with match: {frac:.3g}')
            plt.savefig(plot_dir+'halo_mass.pdf', dpi=400, bbox_inches='tight')
            plt.close()

            _, ax = plt.subplots(1)
            ax.plot(s_matched_stellar, r_matched_stellar, '.')
            ax.plot([np.min(s_matched_stellar), np.max(s_matched_stellar)],
                    [np.min(s_matched_stellar), np.max(s_matched_stellar)], 'k--')
            ax.set_xlabel('Subfind Stellar mass [$M_\odot$]', fontsize=13)
            ax.set_ylabel('Rockstar Stellar mass [$M_\odot$]', fontsize=13)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.savefig(plot_dir+'stellar_mass.pdf', dpi=400, bbox_inches='tight')
            plt.close()

            _, ax = plt.subplots(1)
            ax.plot(s_matched_gas, r_matched_gas, '.')
            ax.plot([np.min(s_matched_gas), np.max(s_matched_gas)],
                    [np.min(s_matched_gas), np.max(s_matched_gas)], 'k--')
            ax.set_xlabel('Subfind Gas mass')
            ax.set_ylabel('Rockstar Gas mass')
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.savefig(plot_dir+'gas_mass.png', dpi=400)
            plt.close()

            _, ax = plt.subplots(1)
            bins = np.arange(8.5, 14.5, 0.5)
            with np.errstate(divide='ignore'):  # Surpress warnings from log10(0)
                ax.hist(np.log10(r_matched_mass), bins=bins, histtype='step',
                        label='Matched halos', density=True)
                ax.hist(np.log10(r_unmatched_mass), bins=bins, histtype='step',
                        label='Unmatched halos', density=True)
            ax.set_xlabel('Halo mass')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of masses for matched and unmatched halos')
            ax.legend()
            plt.savefig(plot_dir+'mass_dist.png', dpi=400)
            plt.close()

            _, ax = plt.subplots(1)
            bins = np.arange(6.5, 11.5, 0.5)
            with np.errstate(divide='ignore'):  # Surpress warnings from log10(0)
                ax.hist(np.log10(r_matched_stellar), bins=bins, histtype='step',
                        label='Matched halos', density=True)
                ax.hist(np.log10(r_unmatched_stellar), bins=bins, histtype='step',
                        label='Unmatched halos', density=True)
            ax.set_xlabel('Stellar mass')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of masses for matched and unmatched halos')
            ax.legend()
            plt.savefig(plot_dir+'stellar_dist.png', dpi=400)
            plt.close()

    # https://physics.stackexchange.com/a/559650/180586
    fig, axs = plt.subplots(2, dpi=300)
    for run_name in run_names[:10]:
        log(f'Loading ages and redshifts for {sim}/{run_name}')
        generated_data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/{run_name}/'
        with open(generated_data_dir+'redshifts.yaml', 'r') as yaml_file:
            redshifts = yaml.safe_load(yaml_file)
        with open(generated_data_dir+'ages.yaml', 'r') as yaml_file:
            ages = yaml.safe_load(yaml_file)
        axs[0].plot(range(34), [redshifts[i] for i in range(34)])
        axs[1].plot(range(34), [ages[i] for i in range(34)])
    axs[0].set_xlabel('Snapshot')
    axs[0].set_ylabel('z')
    axs[1].set_xlabel('Snapshot')
    axs[1].set_ylabel('Ages')
    plt.tight_layout()
    plt.savefig(f'/home/rmcg/camels_matching/{sim}_redshift_range.png')
    plt.close()

# TODO: For each set of cosmological parameters plot n_subfind_Illustris vs n_subfind_simba

# This data is taken from compare/fraction_trackable using TNG100-2
snaps = [4, 10, 18, 33]
redshifts = [3, 2, 1, 0]
tng_mass_cuts = {
    10**9: {
        'frac_has_merger_tree': [0.44, 0.56, 0.7, 1],
        'frac_has_stellar_mass': [0.08, 0.11, 0.13, 0.13],
    },
    10**10: {
        'frac_has_merger_tree': [0.95, 0.98, 0.99, 1],
        'frac_has_stellar_mass': [0.31, 0.4, 0.47, 0.53],
    },
}
for sim in ['IllustrisTNG', 'SIMBA']:
    log(f'Tracking {sim} halos')
    data_dir = f'{base_dir}generated/baryon_tree_ml/camels/{sim}/'
    run_names = sorted(os.listdir(data_dir))
    frac_has_merger_tree, frac_has_stellar_mass = {}, {}
    for mass_cut in tng_mass_cuts:
        frac_has_merger_tree[mass_cut] = {snap: [] for snap in snaps}
        frac_has_stellar_mass[mass_cut] = {snap: [] for snap in snaps}
    for run_name in run_names:
        histories = pd.read_pickle(f'{data_dir}/{run_name}/histories.pickle')
        for mass_cut in tng_mass_cuts:
            mass_cut_histories = histories[histories['33dm_sub_mass'] > mass_cut]
            for snap in snaps:
                merger_tree_frac = np.sum(mass_cut_histories['lowest_snap'] <= snap) / mass_cut_histories.shape[0]
                frac_has_merger_tree[mass_cut][snap].append(merger_tree_frac)
                stellar_mass_frac = np.sum(mass_cut_histories[str(snap)+'stellar_mass'] != 0)
                stellar_mass_frac /= mass_cut_histories.shape[0]
                frac_has_stellar_mass[mass_cut][snap].append(stellar_mass_frac)

    for mass_cut in tng_mass_cuts:
        _, ax = plt.subplots(1, dpi=400)
        sorted_args = np.argsort(frac_has_stellar_mass[mass_cut][33])
        for i_snap, snap in enumerate(snaps):
            p = ax.plot(range(len(run_names)), np.array(frac_has_merger_tree[mass_cut][snap])[sorted_args],
                        '-', label=f'$z={redshifts[i_snap]}$')
            ax.plot(range(len(run_names)), np.array(frac_has_stellar_mass[mass_cut][snap])[sorted_args],
                    '--', color=p[0].get_color())
            ax.plot(range(len(run_names)),
                    tng_mass_cuts[mass_cut]['frac_has_merger_tree'][i_snap]*np.ones(len(run_names)),
                    '-', color=p[0].get_color(), alpha=0.5)
            ax.plot(range(len(run_names)),
                    tng_mass_cuts[mass_cut]['frac_has_stellar_mass'][i_snap]*np.ones(len(run_names)),
                    '--', color=p[0].get_color(), alpha=0.5)

        ax.plot([0], [0], 'k-', label='Fraction with merger trees')
        ax.plot([0], [0], 'k--', label='Fraction with stellar mass')
        ax.plot([0], [0], 'r-', label='Camels')
        ax.plot([0], [0], 'r-', label='TNG100-2', alpha=0.5)
        ax.set_ylim(-0.15, 1.05)
        ax.legend(ncol=4, loc=(0.015, 0.01), fontsize=8.5)
        ax.set_xlabel('Simulation (sorted by frac stellar mass at $z=0$)')
        ax.set_ylabel('Fraction')
        ax.set_title(f'Halo tracking information, $z=0$ mass cut: {mass_cut:.2g}')
        if not os.path.exists('/home/rmcg/camels_matching'):
            os.makedirs('/home/rmcg/camels_matching')
        plt.savefig(f'/home/rmcg/camels_matching/{sim}_trackable_{mass_cut:.2g}.png')
        plt.close()

log('Job finished')
