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
    'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass'],
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass']
}

for sim in ['IllustrisTNG', 'SIMBA']:
    log(f'Running for {sim}')
    data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/'
    run_names = sorted(os.listdir(data_dir))
    for run_name in run_names:
        feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
        with open(feature_importance_dir+'output_features.yaml', 'r') as yaml_file:
            run_output_features = yaml.safe_load(yaml_file)
        assert all([x in output_features.items() for x in run_output_features.items()])
        assert all(snapshots == np.load(feature_importance_dir+'snapshots.npy'))

    params_filename = f'{base_dir}downloaded/camels/{sim}_params.txt'
    params = {  # param: [low_value, high_value, label]
        'omega_m': [0.1, 0.5, '$\Omega_m$'],
        'sigma_8': [0.6, 1, '$\sigma_8$'],
        'a_sn1': [-2, 2, '$\log_2 \:\: A_{SN1}$'],
        'a_agn1': [-2, 2, '$\log_2 \:\: A_{AGN1}$'],
        'a_sn2': [-1, 1, '$\log_2 \:\: A_{SN2}$'],
        'a_agn2': [-1, 1, '$\log_2 \:\: A_{AGN2}$']
    }
    run_params = {param: {} for param in params}
    with open(params_filename, 'r') as params_file:
        lines = [line.strip() for line in params_file]
        for line in lines:
            run_name = line.split(' ')[0]
            run_params['omega_m'][run_name] = float(line.split(' ')[1])
            run_params['sigma_8'][run_name] = float(line.split(' ')[2])
            run_params['a_sn1'][run_name] = np.log2(float(line.split(' ')[3]))
            run_params['a_agn1'][run_name] = np.log2(float(line.split(' ')[4]))
            run_params['a_sn2'][run_name] = np.log2(float(line.split(' ')[5]))
            run_params['a_agn2'][run_name] = np.log2(float(line.split(' ')[6]))

    all_run_importances = [[] for run_name in run_names]
    for i_output, (output_feature, input_properties) in enumerate(output_features.items()):
        log(f'Running PCA for feature importance from predicting {output_feature}')
        plot_dir = f'/home/rmcg/camels_pca/{sim}/{output_feature}/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        run_importances = []
        for i_run, run_name in enumerate(run_names):
            feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
            mean_importance = np.load(f'{feature_importance_dir}{output_feature}_mean_importance.npy')
            run_importances.append(mean_importance)
            all_run_importances[i_run] = all_run_importances[i_run] + mean_importance.tolist()
        run_importances = np.array(run_importances)

        # Use this to apply PCA transfrom from IllustrisTNG to SIMBA
        # Saved model is also used in component_plot.py
        pca_model_filepath = f'/home/rmcg/camels_pca/IllustrisTNG/{output_feature}/pca_model.joblib'
        pca_model_filepath = ''
        if pca_model_filepath:
            pca = joblib.load(pca_model_filepath)
            embedding = pca.transform(run_importances)
        else:
            # Changing n_component has no effect (except giving you more components)
            pca = sklearn.decomposition.PCA(n_components=5)
            embedding = pca.fit_transform(run_importances)
            joblib.dump(pca, plot_dir+'pca_model.joblib')

        # See component_plot.py for connected version of this plot
        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        for component_name, component_values in [
            ('mean', pca.mean_),
            ('component_1', pca.components_[0]),
            ('component_2', pca.components_[1]),
            ('component_3', pca.components_[2]),
            ]:
            fig, ax = plt.subplots(1, dpi=200)
            for input_property in input_properties:
                # TODO: Could estimate std by bootstrapping, running PCA on 90% of camels simulations
                mean_values, sem_values = [], []
                for snap in snapshots:
                    idx = input_features.index(str(snap)+input_property)
                    mean_values.append(component_values[idx])
                mean_values, sem_values = np.array(mean_values), np.array(sem_values)

                ax.plot(snapshots, mean_values, '-o',
                        label=helpers.Config.get_proper_name(input_property, False),
                        markersize=2, color=helpers.Config.get_color(input_property))

            ax.set_ylabel('Feature importance')
            ax.legend(ncol=2, fontsize=7, loc='upper right', handlelength=3)

            padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
            ax.set_xlim([np.min(snapshots)-padding, np.max(snapshots)+padding])
            xticks = np.linspace(np.min(snapshots), np.max(snapshots), 6, dtype=int)
            xticks = np.round(xticks, 1)
            ax.set_xticks(xticks)
            ax.set_xlabel('Snapshot')

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xticklabels([2.63, 1.86, 1.25, 0.69, 0.33, 0])
            ax2.set_xlabel('z')

            title = f'Predict z=0 {helpers.Config.get_proper_name(output_feature, False)}'
            ax.set_title(title)
            plt.tight_layout()

            plot_file = f'{plot_dir}{component_name}_importance'
            plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
            plt.savefig(plot_file+'.png', dpi=150, bbox_inches='tight')
            plt.close()

        i_component_plot = {}
        for param, (low_value, high_value, label) in params.items():
            # Scatter plot of PCA
            fig, ax = plt.subplots(1, dpi=250)
            norm = matplotlib.colors.Normalize(vmin=low_value, vmax=high_value)
            ticks = np.linspace(low_value, high_value, 5)
            cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.rainbow)
            cmap.set_array([])
            fig.colorbar(cmap, label=label, ticks=ticks)

            for i, run_name in enumerate(run_names):
                ax.scatter(embedding[i, 0], embedding[i, 1], color=cmap.to_rgba(run_params[param][run_name]))
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            title = f'PCA applied to FI vectors from predicting {helpers.Config.get_proper_name(output_feature, False)}'
            ax.set_title(title)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            plot_file = f'{plot_dir}scatter_{param}'
            plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
            plt.savefig(plot_file+'.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Averaged version of scatter plot
            fig, ax = plt.subplots(1, dpi=250)
            n_bins = 10
            grid_x = np.linspace(xlim[0], xlim[1], n_bins+1)
            grid_y = np.linspace(ylim[0], ylim[1], n_bins+1)
            nn = np.zeros((n_bins, n_bins))
            vv = np.zeros((n_bins, n_bins))
            for i_run, run_name in enumerate(run_names):
                i_x, i_y = 0, 0
                while embedding[i_run, 0] > grid_x[i_x+1]:
                    i_x += 1
                while embedding[i_run, 1] > grid_y[i_y+1]:
                    i_y += 1
                nn[i_x, i_y] += 1
                vv[i_x, i_y] += run_params[param][run_name]
            vv[nn != 0] /= nn[nn != 0]

            # Color areas with no values white
            vv[nn == 0] = np.nan

            im = ax.imshow(vv.T, origin='lower',
                           cmap='rainbow', vmin=low_value, vmax=high_value,
                           aspect='auto',
                           extent=(xlim[0], xlim[1], ylim[0], ylim[1]))

            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            title = f'PCA applied to FI vectors from predicting {helpers.Config.get_proper_name(output_feature, False)}'
            # ax.set_title(title)
            ax.axvline(0, color='k', ls='--', alpha=0.7)
            ax.axhline(0, color='k', ls='--', alpha=0.7)
            cbar = fig.colorbar(im, ticks=ticks)
            cbar.set_label(label=label, size=12)
            plot_file = f'{plot_dir}average_{param}'
            plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
            plt.savefig(plot_file+'.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Plot value of each component as a function of CAMELS parameters
            n_component = 3
            n_bins = 7
            bins = np.linspace(low_value, high_value, n_bins+1)
            mids = (bins[:-1] + bins[1:]) / 2
            n_in_bin = np.zeros(n_bins)
            v_bin = np.zeros((n_bins, n_component))
            for i_run, run_name in enumerate(run_names):
                i_bin = 0
                while run_params[param][run_name] > bins[i_bin+1]:
                    i_bin += 1
                n_in_bin[i_bin] += 1
                v_bin[i_bin] += embedding[i_run, :n_component]
            for i_bin in range(n_bins):
                v_bin[i_bin] /= n_in_bin[i_bin]

            i_component_plot[param] = {}
            i_component_plot[param]['mids'] = mids
            for i_component in range(n_component):
                i_component_plot[param][i_component] = v_bin[:, i_component]

        component_description = [  # Use as labels for stellar mass pca
            'Component 1 - Decreasing\nimportance of halo potential',
            'Component 2 - Galaxies\nform later',
            'Component 3 - Stronger BH\nmass-stellar mass relation',
        ]
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True, dpi=250)
        axs = axs.flatten()
        for i_param, param in enumerate(params):
            low_value, high_value, label = params[param]
            mids = i_component_plot[param]['mids']
            for i_component in range(n_component):
                v = i_component_plot[param][i_component]
                # p = axs[i_param].plot(mids, v, '-')[0]  # Need to subtract mean if using TNG components on SIMBA
                p = axs[i_param].plot(mids, v-np.mean(v), '-')[0]
                if i_component + 3 == i_param:            # Remove +3 to plot on upper row (change legend loc to lower)
                    axs[i_param].legend(handles=[p], labels=[component_description[i_component]],
                                        loc='upper center', fontsize=12)
            axs[i_param].set_xlabel(label, fontsize=13)
            axs[i_param].set_xticks(np.linspace(low_value, high_value, 5))
            padding = 0.05 * (high_value - low_value)
            axs[i_param].set_xlim(low_value-padding, high_value+padding)
            if i_param < 3:
                axs[i_param].xaxis.tick_top()
                axs[i_param].xaxis.set_label_position('top')
        axs[0].set_ylabel('Mean value for coefficient\nof $i^{th}$ component', fontsize=12)
        axs[3].set_ylabel('Mean value for coefficient\nof $i^{th}$ component', fontsize=12)
        fig.subplots_adjust(wspace=0, hspace=0)
        plot_file = f'{plot_dir}coefficients'
        plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
        plt.savefig(plot_file+'.png', dpi=150, bbox_inches='tight')
        plt.close()

    all_run_importances = [np.array(arr) for arr in all_run_importances]
    pca = sklearn.decomposition.PCA(n_components=2)
    embedding = pca.fit_transform(all_run_importances)

    # TODO: Train model to predict all properties simultaneously, run PCA on FI from it
    for param, (low_value, high_value, label) in params.items():
        plot_dir = f'/home/rmcg/camels_pca/{sim}/all/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, ax = plt.subplots(1, dpi=250)
        norm = matplotlib.colors.Normalize(vmin=low_value, vmax=high_value)
        ticks = np.linspace(low_value, high_value, 5)
        cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.rainbow)
        cmap.set_array([])
        fig.colorbar(cmap, label=label, ticks=ticks)

        for i, run_name in enumerate(run_names):
            plt.scatter(embedding[i, 0], embedding[i, 1], color=cmap.to_rgba(run_params[param][run_name]))
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        title = f'PCA applied to concatenation of all FI vectors'
        ax.set_title(title)
        plot_file = f'{plot_dir}scatter_{param}'
        plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
        plt.savefig(plot_file+'.png', dpi=150, bbox_inches='tight')
        plt.close()

log('Script finished')
