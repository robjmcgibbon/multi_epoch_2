import datetime

import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import sklearn.metrics

import helpers
from helpers import log

log(f'Running {__file__}')
config = helpers.Config('config.yaml')
ages = config.get_ages()
redshifts = config.get_redshifts()

snapshots = config.get_standard_spacing()
# snapshots = config.get_snapshots_within_n_gyr(2)
# snapshots = config.get_standard_spacing_one_snapshot_early()
# snapshots = config.get_tight_spacing()
# snapshots = config.get_every_snapshot()
snapshot_ages = [ages[snap] for snap in snapshots]

data = config.load_data(snapshots)

output_features = {
    'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass'],
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    # 'bh_mass': ['gas_mass', 'dm_sub_mass', 'stellar_mass',
    #     'merge_gas_mass', 'merge_dm_sub_mass', 'merge_stellar_mass'],
    # 'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_stellar_mass'],
    # 'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass', 'merge_stellar_mass'],
    # 'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass'],
    # 'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass',
    #     'merge_bh_mass', 'merge_dm_sub_mass', 'merge_gas_mass', 'merge_stellar_mass'],
}

for output_feature, input_properties in output_features.items():
    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]

    log(f'Training models to predict {output_feature}')
    log(f'Fraction clf != 0: {np.sum(data["clf_"+output_feature])/data.shape[0]:.3g}')
    log(f'Train volume fraction: {config.train_volume_fraction}')

    train_mse = np.zeros(config.n_dataset)
    test_mse = np.zeros(config.n_dataset)
    for i_dataset in range(config.n_dataset):
        log(f'Generating and training for dataset {i_dataset+1}/{config.n_dataset}')
        rf_train, rf_test = config.generate_random_training_box(data, config.train_volume_fraction)

        rf_train = rf_train[rf_train['clf_'+output_feature]]
        rf_test = rf_test[rf_test['clf_'+output_feature]]
        y_train = rf_train['regr_'+output_feature]
        y_test = rf_test['regr_'+output_feature]

        X_train = rf_train[input_features]
        X_test = rf_test[input_features]

        regr = sklearn.ensemble.ExtraTreesRegressor(n_estimators=config.n_estimators,
                                                    n_jobs=config.n_process,
                                                    max_depth=config.get_max_depth(output_feature))
        regr.fit(X_train, y_train)

        train_mse[i_dataset] = sklearn.metrics.mean_squared_error(y_train, regr.predict(X_train))
        y_pred = regr.predict(X_test)
        test_mse[i_dataset] = sklearn.metrics.mean_squared_error(y_test, y_pred)

    log(f'Train MSE: {np.mean(train_mse):.3g} ({np.std(train_mse):.3g})')
    log(f'Test MSE: {np.mean(test_mse):.3g} ({np.std(test_mse):.3g})')

    fig, ax = plt.subplots(1, dpi=150)
    mask = np.random.randint(y_pred.shape[0], size=1000)
    y_test = np.array(y_test)[mask]
    y_pred = np.array(y_pred)[mask]
    ax.plot(y_test, y_pred, '.')
    if output_feature == 'stellar_mass':
        xlabel = r'$\log (M_{\bigstar, \mathrm{True}} \ [M_\odot\!])$'
        ylabel = r'$\log (M_{\bigstar, \mathrm{Predicted}} \ [M_\odot\!])$'
    elif output_feature == 'bh_mass':
        xlabel = r'$\log (M_{BH, \mathrm{True}} \ [M_\odot\!])$'
        ylabel = r'$\log (M_{BH, \mathrm{Predicted}} \ [M_\odot\!])$'
    else:
        # TODO: These labels should be log
        xlabel = f'{config.get_proper_name(output_feature, False)} true value '
        xlabel += config.get_units_with_brackets(output_feature)
        ylabel = f'{config.get_proper_name(output_feature, False)} predicted value'
        ylabel += config.get_units_with_brackets(output_feature)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    # TODO: Density plot???

    plot_name = f'btml_{output_feature}_true_v_pred'
    config.plot_show_save(plot_name, fig, force_save=True)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-config.start_time}\n')
