import os
import sys

import joblib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import yaml

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

base_dir = helpers.Config.get_base_dir()

snapshots = np.arange(6, 34, 3)
output_features = {
    # 'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass'],
    # 'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    # 'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    # 'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass']
}

for output_feature, input_properties in output_features.items():
    pca_model_filepath = f'/home/rmcg/camels_pca/IllustrisTNG/{output_feature}/pca_model.joblib'
    pca = joblib.load(pca_model_filepath)

    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
    fig, axs = plt.subplots(nrows=4, figsize=(5, 12), sharex=True)
    fig.subplots_adjust(hspace=0)
    for i_component, (component_name, component_values, label, yticks) in enumerate([
        ('mean', pca.mean_ / np.max(pca.mean_), 'Mean', [0, 0.2, 0.4, 0.6, 0.8, 1]),  # Note that mean is normalised
        ('component_1', pca.components_[0], 'Component 1', [-0.2, 0, 0.2, 0.4]),
        ('component_2', pca.components_[1], 'Component 2', [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]),
        ('component_3', pca.components_[2], 'Component 3', [-0.2, 0, 0.2, 0.4, 0.6]),
        ]):
        ax = axs[i_component]
        for i_prop, input_property in enumerate(input_properties):
            # TODO: Could estimate std by bootstrapping, running PCA on 90% of camels simulations
            mean_values, sem_values = [], []
            for snap in snapshots:
                idx = input_features.index(str(snap)+input_property)
                mean_values.append(component_values[idx])
            mean_values, sem_values = np.array(mean_values), np.array(sem_values)

            p = ax.plot(snapshots, mean_values, '-o',
                    markersize=2, color=helpers.Config.get_color(input_property))[0]
            if i_component == i_prop:
                ax.legend(handles=[p], labels=[helpers.Config.get_proper_name(input_property, False)],
                          loc='upper right', fontsize=12)

        padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
        ax.set_xlim([np.min(snapshots)-padding, np.max(snapshots)+padding])
        xticks = np.linspace(np.min(snapshots), np.max(snapshots), 6, dtype=int)
        xticks = np.round(xticks, 1)
        ax.set_xticks(xticks)

        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.25)
        ax.set_yticks(yticks)

        ax.text(0.05, 0.88, label, fontsize=12, transform=ax.transAxes)

    axs[0].set_ylabel('Feature importance')
    for ax in axs[1:]:
        ax.set_ylabel('Difference in\nfeature importance')

    ax2 = axs[0].twiny()
    ax2.set_xlim(axs[0].get_xlim())
    ax2.set_xticks(axs[0].get_xticks())
    ax2.set_xticklabels([2.63, 1.86, 1.25, 0.69, 0.33, 0])
    ax2.set_xlabel('z')

    axs[-1].set_xlabel('Snapshot')

    plt.savefig('/home/rmcg/test.pdf', dpi=450, bbox_inches='tight')
    plt.close()

log('Script finished')
