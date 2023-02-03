import os
import sys

import joblib
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
    # Checking saved data is consistent with snapshots and features specified above
    data_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/{sim}/'
    run_names = sorted(os.listdir(data_dir))
    for run_name in run_names:
        feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
        with open(feature_importance_dir+'output_features.yaml', 'r') as yaml_file:
            run_output_features = yaml.safe_load(yaml_file)
        assert all([x in output_features.items() for x in run_output_features.items()])
        assert all(snapshots == np.load(feature_importance_dir+'snapshots.npy'))

    params = {  # param: [low_value, high_value, label]
        'omega_m': [0.1, 0.5, '$\Omega_m$'],
        'sigma_8': [0.6, 1, '$\sigma_8$'],
        'a_sn1': [-2, 2, '$\log_2 \:\: A_{SN1}$'],
        'a_agn1': [-2, 2, '$\log_2 \:\: A_{AGN1}$'],
        'a_sn2': [-1, 1, '$\log_2 \:\: A_{SN2}$'],
        'a_agn2': [-1, 1, '$\log_2 \:\: A_{AGN2}$']
    }
    run_params = {param: {} for param in params}
    run_params_filename = f'{base_dir}downloaded/camels/{sim}_params.txt'
    with open(run_params_filename, 'r') as params_file:
        lines = [line.strip() for line in params_file]
        for line in lines:
            run_name = line.split(' ')[0]
            if run_name not in run_names:
                continue
            run_params['omega_m'][run_name] = float(line.split(' ')[1])
            run_params['sigma_8'][run_name] = float(line.split(' ')[2])
            run_params['a_sn1'][run_name] = np.log2(float(line.split(' ')[3]))
            run_params['a_agn1'][run_name] = np.log2(float(line.split(' ')[4]))
            run_params['a_sn2'][run_name] = np.log2(float(line.split(' ')[5]))
            run_params['a_agn2'][run_name] = np.log2(float(line.split(' ')[6]))

    # Creating arrays as input to interpolation
    run_params = np.array([[run_params[p][r] for p in params] for r in run_names])

    for i_output, (output_feature, input_properties) in enumerate(output_features.items()):
        log(f'Interpolating feature importance from predicting {output_feature}')
        interpolate_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/interpolate/{sim}/'
        if not os.path.exists(interpolate_dir):
            os.makedirs(interpolate_dir)

        run_importances = []
        for i_run, run_name in enumerate(run_names):
            feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
            mean_importance = np.load(f'{feature_importance_dir}{output_feature}_mean_importance.npy')
            run_importances.append(mean_importance)
        run_importances = np.array(run_importances)

        # TODO: Use scipy for interpolation? Other ML algorithms? min_samples_leaf value?
        interpolator_name = f'{output_feature}_interpolator.joblib'
        interpolator = sklearn.ensemble.RandomForestRegressor(n_jobs=5, min_samples_leaf=4)
        interpolator.fit(run_params, run_importances)
        joblib.dump(interpolator, interpolate_dir+interpolator_name)

        with open(interpolate_dir+'output_features.yaml', 'w') as yaml_file:
            yaml.dump(output_features, yaml_file)
        np.save(interpolate_dir+'snapshots.npy', snapshots)

        n_plot = 20
        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        for i_run in np.random.randint(0, len(run_names), n_plot):
            run_name = run_names[i_run]
            feature_importance_dir = f'{data_dir}{run_name}/feature_importance/'
            mean_importance = np.load(f'{feature_importance_dir}{output_feature}_mean_importance.npy')
            sem_importance = np.load(f'{feature_importance_dir}{output_feature}_sem_importance.npy')
            pred_importance = interpolator.predict(run_params[i_run].reshape(1, -1))[0]

            fig, ax = plt.subplots(1, dpi=200)
            for input_property in input_properties:
                mean_values, sem_values, pred_values = [], [], []
                for snap in snapshots:
                    idx = input_features.index(str(snap)+input_property)
                    mean_values.append(mean_importance[idx])
                    sem_values.append(sem_importance[idx])
                    pred_values.append(pred_importance[idx])
                mean_values = np.array(mean_values)
                sem_values = np.array(sem_values)
                pred_values = np.array(pred_values)

                ax.plot(snapshots, mean_values, 'o',
                        linestyle='solid',
                        label=helpers.Config.get_proper_name(input_property, False),
                        markersize=2, color=helpers.Config.get_color(input_property))

                ax.plot(snapshots, pred_values, 'o',
                        linestyle='dashed',
                        markersize=2, color=helpers.Config.get_color(input_property))

                ax.fill_between(snapshots, mean_values-sem_values, mean_values+sem_values,
                                color=helpers.Config.get_color(input_property), edgecolor='w', alpha=0.2)

            ax.plot([0], [0], 'k-', label='Simulation')
            ax.plot([0], [0], 'k--', label='Interpolation')
            ax.set_ylim(0)
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

            ax.set_title(f'Predict z=0 {helpers.Config.get_proper_name(output_feature, False)}')
            plt.tight_layout()

            plot_dir = f'/home/rmcg/camels_interpolation/{sim}/{output_feature}/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(f'{plot_dir}{run_name}.png')
            plt.close()

log('Script finished')
