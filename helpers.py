import argparse
import datetime
import os
import socket
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble


# TODO: It could be worth setting the random seed in here to ensure consistent plots

def log(*message):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, *message)


class NoDataError(Exception):
    pass


class Config:

    def __init__(self, config_filename, is_main_config=True):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type=str)
        parser, _ = parser.parse_known_args()
        if parser.config_file:
            config_filename = parser.config_file

        with open(config_filename, 'r') as config_file:
            config_default = yaml.safe_load(config_file)
        parser = argparse.ArgumentParser()
        parser.set_defaults(**config_default)
        parser.add_argument('--job_id', type=str)
        parser.add_argument('--n_dataset', type=int)
        parser.add_argument('--n_estimators', type=int)
        parser.add_argument('--n_process', type=int)
        parser.add_argument('--train_volume_fraction', type=float)
        parser.add_argument('--mask', type=int, nargs='*')
        parser.add_argument('--box_size', type=int)
        parser.add_argument('--simulation', type=str)
        parser.add_argument('--tree_algorithm', type=str)
        parser.add_argument('--redshift_to_predict', type=int)
        parser.add_argument('--run_number', type=int)

        parser, unknown = parser.parse_known_args()
        if unknown:
            log(f'Unknown arguments: {unknown}')
        parser = vars(parser)

        # Main and secondary configs will get these
        self.n_estimators = parser['n_estimators']
        self.n_dataset = parser['n_dataset']
        self.n_process = parser['n_process']
        self.train_volume_fraction = parser['train_volume_fraction']

        self.simulation = parser['simulation']
        self.box_size = parser['box_size']
        self.tree_algorithm = parser['tree_algorithm']
        self.run_number = parser['run_number']
        self.redshift_to_predict = parser['redshift_to_predict']
        self.mask = parser['mask']
        self.mask_desc = ''
        self.name = config_default.get('name', '_')

        if is_main_config:
            self.start_time = datetime.datetime.now()
            self.plot_dir = os.path.expanduser('~') + '/'
            if parser['job_id']:
                self.plot_dir += 'job_' + parser['job_id'] + '/'
                if not os.path.exists(self.plot_dir):
                    os.makedirs(self.plot_dir)
            log(f'Hostname: {socket.gethostname()}')
            self.finish_init()

    @staticmethod
    def load_secondary_configs(config_filename):
        secondary_configs = []
        with open(config_filename, 'r') as config_file:
            config_default = yaml.safe_load(config_file)
        for secondary_config in config_default['secondary_configs']:
            config = Config(config_filename, is_main_config=False)

            config.simulation = secondary_config['simulation']
            config.box_size = secondary_config['box_size']
            config.tree_algorithm = secondary_config['tree_algorithm']
            config.run_number = secondary_config['run_number']
            config.redshift_to_predict = secondary_config['redshift_to_predict']
            config.mask = secondary_config['mask']
            config.name = secondary_config.get('name', '_')

            config.finish_init()
            secondary_configs.append(config)
        return secondary_configs

    # noinspection PyAttributeOutsideInit
    def finish_init(self):
        # Set box_length, which is size in comoving Mpc. Box size must be valid, or KeyError is thrown
        if self.simulation in ['tng', 'illustris']:
            self.box_length = {
                50: 35 / self.get_hubble_constant(),
                100: 75 / self.get_hubble_constant(),
                300: 205 / self.get_hubble_constant(),
            }[self.box_size]
        elif self.simulation.split('_')[0] == 'eagle':
            if self.simulation == 'eagle_ref':
                assert self.box_size in [50, 100]
            else:
                assert self.box_size == 50
            self.box_length = self.box_size
        elif (self.simulation.split('_')[0] == 'galform') or (self.simulation in ['lgalaxies_2010', 'lgalaxies_2013']):
            assert self.box_size == 100
            raise NotImplementedError
        elif self.simulation == 'lgalaxies_2020':
            assert self.tree_algorithm == 'lhalotree'
            assert self.run_number == 1
            self.box_length = {
                100: 75 / self.get_hubble_constant(),
                300: 205 / self.get_hubble_constant(),
            }[self.box_size]
        elif self.simulation.split('_')[0] == 'simba':
            assert self.box_size == 100
            raise NotImplementedError
        else:
            raise NotImplementedError

        # TODO: I'm not happy about how this is done
        snapshot = {'tng': {0: 99, 1: 50, 2: 33},
                    'illustris': {0: 135, 1: 85, 2: 68},
                    'galform_2006': {0: 63, 1: 41, 2: 32},
                    'galform_2014': {0: 61, 1: 39, 2: 30},
                    'eagle': {0: 28, 1: 19, 2: 15},
                    'simba': {0: 150, 1: 10.5, 2: 78},
                    }
        snapshot['lgalaxies_2020'] = snapshot['tng']
        snapshot['galform_2012'] = snapshot['galform_2006']
        snapshot['galform_2016'] = snapshot['galform_2014']
        snapshot['lgalaxies_2010'] = snapshot['galform_2006']
        snapshot['lgalaxies_2013'] = snapshot['galform_2014']
        if self.simulation in ['eagle_HiML', 'eagle_LoML']:
            self.snapshot_to_predict = {0: 10}[self.redshift_to_predict]
        elif self.simulation.split('_')[0] == 'eagle':
            self.snapshot_to_predict = snapshot['eagle'][self.redshift_to_predict]
        elif self.simulation.split('_')[0] == 'simba':
            self.snapshot_to_predict = snapshot['simba'][self.redshift_to_predict]
        else:
            self.snapshot_to_predict = snapshot[self.simulation][self.redshift_to_predict]

        if self.name != '_':
            pass
        elif self.simulation == 'tng':
            self.name = f'TNG{self.box_size}-{self.run_number}'
        elif self.simulation == 'illustris':
            self.name = f'Illustris{self.run_number}'
        elif self.simulation.split('_')[0] == 'eagle':
            self.name = f'{self.simulation}-{self.box_size}'
        elif self.simulation == 'lgalaxies_2020':
            self.name = f'lgalaxies2020-{self.box_size}'
        elif (self.simulation.split('_')[0] == 'galform') or (self.simulation in ['lgalaxies_2010', 'lgalaxies_2013']):
            self.name = f'{"".join(self.simulation.split("_"))}-{self.box_size}'
        else:
            raise NotImplementedError

        log(f'Config: {self}')

    def __repr__(self):
        return str(vars(self))

    @staticmethod
    def get_base_dir():
        hostname = socket.gethostname()
        if hostname == 'lenovo-p52':
            return os.path.expanduser('~') + '/data/'
        else:
            return '/disk01/rmcg/'

    def get_raw_data_dir(self):
        data_dir = self.get_base_dir()
        if self.simulation == 'tng':
            data_dir += f'downloaded/tng/tng{self.box_size}-{self.run_number}/'
            data_dir += f'merger_tree/{self.tree_algorithm}/'
        elif self.simulation == 'illustris':
            data_dir += f'downloaded/illustris/illustris-{self.run_number}/'
            data_dir += f'merger_tree/{self.tree_algorithm}/'
        elif self.simulation == 'lgalaxies_2020':
            data_dir += f'downloaded/tng/tng{self.box_size}-{self.run_number}-dark/'
            data_dir += f'merger_tree/{self.tree_algorithm}/'
        elif self.simulation.split('_')[0] == 'simba':
            data_dir += f'downloaded/simba/m50n512/'
            data_dir += self.simulation.split('_')[1] + '/'
        else:
            raise NotImplementedError
        return data_dir

    def get_generated_data_dir(self):
        data_dir = self.get_base_dir() + 'generated/baryon_tree_ml/'
        if self.simulation == 'tng':
            data_dir += f'tng{self.box_size}-{self.run_number}/{self.tree_algorithm}/z_{self.redshift_to_predict}/'
        elif self.simulation == 'illustris':
            data_dir += f'illustris-{self.run_number}/{self.tree_algorithm}/z_{self.redshift_to_predict}/'
        elif self.simulation == 'lgalaxies_2020':
            data_dir += f'lgalaxies_2020_{self.box_size}/{self.tree_algorithm}/z_{self.redshift_to_predict}/'
        elif (self.simulation.split('_')[0] == 'galform') or (self.simulation in ['lgalaxies_2010', 'lgalaxies_2013']):
            data_dir += f'{self.simulation}_{self.box_size}/z_{self.redshift_to_predict}/'
        elif self.simulation.split('_')[0] == 'eagle':
            data_dir += f'{self.simulation}{self.box_size}/z_{self.redshift_to_predict}/'
        elif self.simulation.split('_')[0] == 'simba':
            data_dir += f'{self.simulation}/z_{self.redshift_to_predict}/'
        else:
            raise NotImplementedError
        return data_dir

    def load_data(self, snapshots):
        generated_data_dir = self.get_generated_data_dir()
        data = pd.read_pickle(generated_data_dir + 'data.pickle')

        mask_definitions = {
            0: (np.ones(data.shape[0], dtype=bool),
                'All galaxies'),

            1: ((data['stellar_mass'] > 10 ** 9) & (data['stellar_mass'] < 10 ** 10),
                '$10^{9} M_\odot < M_* < 10^{10} M_\odot$'),

            2: ((data['stellar_mass'] > 10 ** 10) & (data['stellar_mass'] < 10 ** 11),
                '$10^{10} M_\odot < M_* < 10^{11} M_\odot$'),

            3: (data['central'] == 0,
                'Satellite galaxies'),

            4: (data['central'] == 1,
                'Central galaxies'),

            5: (data['lowest_snap'] <= np.min(snapshots),
                f'Galaxies which can be tracked to z={self.get_redshifts()[np.min(snapshots)]}'),

            #  Cutoff taken from https://arxiv.org/pdf/2105.05298.pdf, figure 8
            6: ((data['ssfr'] != -1) & (data['ssfr'] < 10 ** -11),
                '$sSFR < 10^{-11} yr^{-1}$'),

            7: ((data['ssfr'] != -1) & (data['ssfr'] > 10 ** -11),
                '$sSFR > 10^{-11} yr^{-1}$'),

            8: ((10 ** 14 < data['fof_mass']) & (data['fof_mass'] < 10 ** 15),
                '$10^{14} M_\odot < M_{FOF} < 10^{15} M_\odot$'),

            9: ((10 ** 13 < data['fof_mass']) & (data['fof_mass'] < 10 ** 14),
                '$10^{13} M_\odot < M_{FOF} < 10^{14} M_\odot$'),

            10: ((10 ** 12 < data['fof_mass']) & (data['fof_mass'] < 10 ** 13),
                 '$10^{12} M_\odot < M_{FOF} < 10^{13} M_\odot$'),

            11: ((data['stellar_mass'] > 10 ** 8) & (data['stellar_mass'] < 10 ** 12),
                '$10^{8} M_\odot < M_* < 10^{12} M_\odot$'),
        }

        data_mask = np.ones(data.shape[0], dtype=bool)
        for i_mask in self.mask:
            new_mask, new_mask_desc = mask_definitions[i_mask]
            data_mask = data_mask & new_mask
            self.mask_desc += new_mask_desc + ', '
        self.mask_desc = self.mask_desc[:-2]

        log(f'Loaded data for {self.name}. Mask used: {self.mask_desc}')
        log(f'Fraction halos kept: {np.sum(data_mask) / data_mask.shape[0]:.2g}')
        log(f'Number halos kept: {np.sum(data_mask)}')
        return data[data_mask]

    def get_mass_limit(self):
        # Used to stop the lowest mass halos being processed, which allows scripts to run faster
        if self.simulation in ['tng', 'illustris', 'lgalaxies_2020']:
            mass_limit = {50: {1: 10**8, 2: 10**9, 3: 10**10},
                          100: {1: 10**9, 2: 10**10, 3: 10**11},
                          300: {1: 10**10, 2: 10**11, 3: 10**12}}
            return mass_limit[self.box_size][self.run_number]
        if (self.simulation.split('_')[0] == 'galform') or (self.simulation in ['lgalaxies_2010', 'lgalaxies_2013']):
            return 10**11
        if self.simulation.split('_')[0] == 'eagle':
            return 10**10

    def get_hubble_constant(self):
        if self.simulation in ['tng', 'lgalaxies_2020']:
            return 0.6774
        if self.simulation == 'illustris':
            return 0.704
        if self.simulation in ['galform_2006', 'galform_2012', 'lgalaxies_2010']:
            return 0.72
        if self.simulation in ['galform_2014', 'galform_2016', 'lgalaxies_2013']:
            return 0.703
        raise NotImplementedError


    def get_min_snap_to_extract(self):
        if self.simulation in ['lgalaxies_2020', 'tng', 'illustris']:
            return 2
        if self.simulation.split('_')[0] == 'eagle':
            return 0
        if self.simulation.split('_')[0] == 'simba':
            return 50
        # TODO: What value should this be?
        if self.simulation.split('_')[0] == 'galform':
            return 17
        raise NotImplementedError

    def get_redshifts(self):
        try:
            return self.redshifts
        except AttributeError:
            with open(self.get_generated_data_dir()+'redshifts.yaml', 'r') as yaml_file:
                # noinspection PyAttributeOutsideInit
                self.redshifts = yaml.safe_load(yaml_file)
            return self.redshifts

    def get_ages(self):
        try:
            return self.ages
        except AttributeError:
            with open(self.get_generated_data_dir()+'ages.yaml', 'r') as yaml_file:
                # noinspection PyAttributeOutsideInit
                self.ages = yaml.safe_load(yaml_file)
            return self.ages

    def get_closest_snapshot_for_redshift(self, redshift):
        min_dist = float('inf')
        closest_snap = 0
        # TODO: Fix
        # for snap in range(min_snap, self.snapshot_to_predict + 1):
        for snap in range(self.snapshot_to_predict + 1):
            # Use get, default to -1
            dist = abs(redshift - self.get_redshifts()[snap])
            if dist < min_dist:
                min_dist = dist
                closest_snap = snap
        return closest_snap

    def get_closest_snapshot_for_age(self, age):
        min_dist = float('inf')
        closest_snap = 0
        # TODO: Fix
        # for snap in range(min_snap, self.snapshot_to_predict + 1):
        for snap in range(self.snapshot_to_predict + 1):
            # Use get, default to -1
            dist = abs(age - self.get_ages()[snap])
            if dist < min_dist:
                min_dist = dist
                closest_snap = snap
        return closest_snap

    @staticmethod
    def get_lookback_time(redshift):
        # Values taken from https://home.fnal.gov/~gnedin/cc/
        lookback_time = {0: 0, 1: 7.9, 2: 10.5, 3: 11.6, 4: 12.3, 5: 12.6, 6: 12.8, 7: 13}
        return lookback_time[redshift]

    def get_standard_spacing(self):
        if self.simulation in ['eagle_HiML', 'eagle_LoML']:
            return np.arange(1, 11)
        max_age = self.get_ages()[self.snapshot_to_predict]
        # TODO: What should the start point be?
        age_snapshots = np.linspace(1, max_age, 10)
        snapshots = [self.get_closest_snapshot_for_age(age) for age in age_snapshots]
        return np.array(snapshots)

    def get_snapshots_within_n_gyr(self, n):
        max_age = self.get_ages()[self.snapshot_to_predict]
        snapshots = [snap for snap in self.get_every_snapshot() if self.get_ages()[snap] > max_age - n]
        return np.array(sorted(snapshots))

    def get_standard_spacing_one_snapshot_early(self):
        if self.simulation in ['eagle_HiML', 'eagle_LoML']:
            raise NotImplementedError
        spacing = np.array(self.get_standard_spacing())
        return spacing - 1

    def get_tight_spacing(self):
        max_age = self.get_ages()[self.snapshot_to_predict]
        age_snapshots = np.linspace(0.7, max_age, 20)
        snapshots = [self.get_closest_snapshot_for_age(age) for age in age_snapshots]
        return np.array(snapshots)

    def get_every_snapshot(self):
        min_snap = np.min(self.get_standard_spacing())
        return np.arange(min_snap, self.snapshot_to_predict+1)

    def plot_show_save(self, plot_name, fig, force_save=False, close=True, data=None):
        metadata = {f'config_{k}': str(v) for k, v in self.__dict__.items()}
        if not data:
            data = {}
            for i_ax, ax in enumerate(fig.get_axes()):
                for i_line, line in enumerate(ax.lines):
                    data[line.get_color()] = line.get_data()
        metadata['config_data'] = str(data).replace('\n', '')

        hostname = socket.gethostname()
        if (hostname == 'lenovo-p52') and (not force_save):
            plt.show()
        else:
            if not os.path.exists(self.plot_dir + os.path.dirname(plot_name)):
                os.makedirs(self.plot_dir + os.path.dirname(plot_name))
            plot_file = self.plot_dir + plot_name
            plt.savefig(plot_file+'.pdf', dpi=450, bbox_inches='tight')
            plt.savefig(plot_file+'.png', dpi=150, metadata=metadata, bbox_inches='tight')
        if close:
            plt.close()

    def add_redshift_labels(self, ax):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())

        ax2_xlabels = []
        for age in ax.get_xticks():
            closest_snap = self.get_closest_snapshot_for_age(age)
            closest_redshift = self.get_redshifts()[closest_snap]
            closest_redshift = round(closest_redshift, 1)
            ax2_xlabels.append(closest_redshift)
        ax2.set_xticklabels(ax2_xlabels)
        ax2.set_xlabel('z')
        return list(ax.get_xlim()), ax.get_xticks().tolist(), ax2_xlabels

    @staticmethod
    def add_unique_legend(ax):
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

    @staticmethod
    def get_color(input_property):
        # Hex color codes of matplotlib tab: named colors, found using color picker
        color = {'bh_dot': '#17BECF',              # Cyan
                 'merge_bh_dot': '#17BECF',        # Cyan
                 'bh_mass': '#1F77B4',             # Blue
                 'merge_bh_mass': '#1F77B4',       # Blue
                 'cold_gas': '#0b5b0b',            # Dark green
                 'dm_fof_mass': '#7F7F7F',         # Grey
                 'merge_dm_fof_mass': '#7F7F7F',   # Grey
                 'dm_sub_mass': '#7F7F7F',         # Grey
                 'merge_dm_sub_mass': '#7F7F7F',   # Grey
                 'dm_mass': '#7F7F7F',             # Grey
                 'gas_mass': '#2BA02B',            # Green
                 'merge_gas_mass': '#2BA02B',      # Green
                 'hot_gas': '#95cf95',             # Light green
                 'sfr': 'yellow',
                 'ssfr': 'olive',
                 'stellar_mass': '#FF7F0E',        # Orange
                 'stellar_metallicity': 'purple',
                 'merge_stellar_mass': '#FF7F0E'}  # Orange
        return color[input_property]

    @staticmethod
    def get_proper_name(property, include_units):
        labels = {'bh_dot': 'BH accretion rate',
                  'merge_bh_dot': 'Merger BH accretion rate',
                  'bh_mass': 'BH mass',
                  'merge_bh_mass': 'Merger BH mass',
                  'cold_gas': 'Cold gas mass',
                  'dm_fof_mass': 'FOF DM mass',
                  'merge_dm_fof_mass': 'Merger FOF DM mass',
                  'dm_sub_mass': 'Subhalo DM mass',
                  'dm_mass': 'DM mass',
                  'merge_dm_sub_mass': 'Merger subhalo DM mass',
                  'hot_gas': 'Hot gas mass',
                  'gas_mass': 'Gas mass',
                  'merge_gas_mass': 'Merger gas mass',
                  'mock_g': 'g band',
                  'mock_k': 'K band',
                  'mock_u': 'U band',
                  'sfr': 'Star formation rate',
                  'ssfr': 'Specific star formation rate',
                  'stellar_mass': 'Stellar mass',
                  'merge_stellar_mass': 'Merger stellar mass',
                  'stellar_metallicity': 'Stellar metallicity'}
        if include_units:
            raise NotImplementedError
            # Use function below to get units
            # I couldn't be bothered changing all the code which has the include_units argument
        return labels[property]

    @staticmethod
    def get_units_with_brackets(property):
        labels = {'bh_dot': '$M_\odot$ / yr',
                  'merge_bh_dot': '$M_\odot$ / yr',
                  'bh_mass': '$M_\odot$',
                  'merge_bh_mass': '$M_\odot$',
                  'cold_gas': '$M_\odot$',
                  'dm_fof_mass': '$M_\odot$',
                  'merge_dm_fof_mass': '$M_\odot$',
                  'dm_sub_mass': '$M_\odot$',
                  'dm_mass': '$M_\odot$',
                  'merge_dm_sub_mass': '$M_\odot$',
                  'hot_gas': '$M_\odot$',
                  'gas_mass': '$M_\odot$',
                  'merge_gas_mass': '$M_\odot$',
                  'mock_g': 'mag',
                  'mock_k': 'mag',
                  'mock_u': 'mag',
                  'sfr': '$M_\odot$ / yr',
                  'ssfr': r'\textrm{yr}^{-1}',
                  'stellar_mass': '$M_\odot$',
                  'merge_stellar_mass': '$M_\odot$',
                  'stellar_metallicity': ''}
        if property in ['stellar_metallicity']:
            return ''
        return '[' + labels[property] + ']'

    # noinspection PyUnusedLocal
    @staticmethod
    def get_max_depth(output_feature):
        return 12

    def generate_random_training_box(self, data, frac):
        # frac gives train_volume / full_volume
        train_box_length = np.cbrt(frac) * self.box_length

        shift = np.random.uniform(low=0, high=self.box_length, size=3)
        pos = np.array(data[['x', 'y', 'z']])
        pos += shift
        pos %= self.box_length

        train_mask = np.all(pos < train_box_length, axis=1)
        test_mask = np.logical_not(train_mask)

        return data[train_mask], data[test_mask]

    def calculate_feature_importance(self, data, input_features, output_feature, train_volume_fraction=None):
        log(f'Training models to predict {output_feature} for {self.name}')
        if np.sum(data["clf_"+output_feature]) == 0:
            raise NoDataError
        log(f'Fraction clf != 0: {np.sum(data["clf_"+output_feature])/data.shape[0]:.3g}')

        if train_volume_fraction is None:
            train_volume_fraction = self.train_volume_fraction

        importances = np.zeros((self.n_dataset, len(input_features)))
        for i_dataset in range(self.n_dataset):
            log(f'Generating and training for dataset {i_dataset+1}/{self.n_dataset}')
            rf_train, rf_test = self.generate_random_training_box(data, train_volume_fraction)

            rf_train = rf_train[rf_train['clf_'+output_feature]]
            rf_test = rf_test[rf_test['clf_'+output_feature]]
            y_train = rf_train['regr_'+output_feature]
            y_test = rf_test['regr_'+output_feature]

            X_train = rf_train[input_features]
            X_test = rf_test[input_features]

            regr = sklearn.ensemble.ExtraTreesRegressor(n_estimators=self.n_estimators,
                                                        n_jobs=self.n_process,
                                                        max_depth=self.get_max_depth(output_feature))
            regr.fit(X_train, y_train)

            importances[i_dataset] = regr.feature_importances_

        mean_importance = np.mean(importances, axis=0)
        sem_importance = 1.96 * np.std(importances, axis=0) / np.sqrt(self.n_dataset)
        return mean_importance, sem_importance


def get_tng_redshifts():
    # Values taken from tng100-1 webpage
    return {0: 20.05, 1: 14.99, 2: 11.98, 3: 10.98, 4: 10.00, 5: 9.39,
            6: 9.00, 7: 8.45, 8: 8.01, 9: 7.60, 10: 7.24, 11: 7.01,
            12: 6.49, 13: 6.01, 14: 5.85, 15: 5.53, 16: 5.23, 17: 5.00,
            18: 4.66, 19: 4.43, 20: 4.18, 21: 4.01, 22: 3.71, 23: 3.49,
            24: 3.28, 25: 3.01, 26: 2.90, 27: 2.73, 28: 2.58, 29: 2.44,
            30: 2.32, 31: 2.21, 32: 2.10, 33: 2.00, 34: 1.90, 35: 1.82,
            36: 1.74, 37: 1.67, 38: 1.60, 39: 1.53, 40: 1.50, 41: 1.41,
            42: 1.36, 43: 1.30, 44: 1.25, 45: 1.21, 46: 1.15, 47: 1.11,
            48: 1.07, 49: 1.04, 50: 1.00, 51: 0.95, 52: 0.92, 53: 0.89,
            54: 0.85, 55: 0.82, 56: 0.79, 57: 0.76, 58: 0.73, 59: 0.70,
            60: 0.68, 61: 0.64, 62: 0.62, 63: 0.60, 64: 0.58, 65: 0.55,
            66: 0.52, 67: 0.50, 68: 0.48, 69: 0.46, 70: 0.44, 71: 0.42,
            72: 0.40, 73: 0.38, 74: 0.36, 75: 0.35, 76: 0.33, 77: 0.31,
            78: 0.30, 79: 0.27, 80: 0.26, 81: 0.24, 82: 0.23, 83: 0.21,
            84: 0.20, 85: 0.18, 86: 0.17, 87: 0.15, 88: 0.14, 89: 0.13,
            90: 0.11, 91: 0.10, 92: 0.08, 93: 0.07, 94: 0.06, 95: 0.05,
            96: 0.03, 97: 0.02, 98: 0.01, 99: 0.00}


def get_illustris_redshifts():
    # Values taken from illustris-2 webpage
    # Note that snapshots 53 and 55 are missing for Illustris-1
    return {0: 46.77, 1: 44.56, 2: 42.45, 3: 40.64, 4: 38.71, 5: 36.87,
            6: 35.12, 7: 33.61, 8: 32.01, 9: 30.48, 10: 29.03, 11: 27.64,
            12: 26.44, 13: 25.17, 14: 23.96, 15: 22.81, 16: 21.81, 17: 20.76,
            18: 19.75, 19: 18.79, 20: 17.96, 21: 17.09, 22: 16.25, 23: 15.45,
            24: 14.76, 25: 14.03, 26: 13.34, 27: 12.67, 28: 12.04, 29: 11.50,
            30: 10.92, 31: 10.37, 32: 10.00, 33: 9.84, 34: 9.39, 35: 9.00,
            36: 8.91, 37: 8.45, 38: 8.01, 39: 7.60, 40: 7.24, 41: 7.01,
            42: 6.86, 43: 6.49, 44: 6.14, 45: 6.01, 46: 5.85, 47: 5.53,
            48: 5.23, 49: 5.00, 50: 4.94, 51: 4.66, 52: 4.43, 53: 4.18, 54: 4.01,
            55: 3.94, 56: 3.71, 57: 3.49, 58: 3.28, 59: 3.08, 60: 3.01, 61: 2.90,
            62: 2.73, 63: 2.58, 64: 2.44, 65: 2.32, 66: 2.21, 67: 2.10,
            68: 2.00, 69: 1.90, 70: 1.82, 71: 1.74, 72: 1.67, 73: 1.60,
            74: 1.53, 75: 1.47, 76: 1.41, 77: 1.36, 78: 1.30, 79: 1.25,
            80: 1.21, 81: 1.15, 82: 1.11, 83: 1.07, 84: 1.04, 85: 1.00,
            86: 0.99, 87: 0.95, 88: 0.92, 89: 0.89, 90: 0.85, 91: 0.82,
            92: 0.79, 93: 0.76, 94: 0.73, 95: 0.70, 96: 0.68, 97: 0.64,
            98: 0.62, 99: 0.60, 100: 0.58, 101: 0.55, 102: 0.52, 103: 0.50,
            104: 0.48, 105: 0.46, 106: 0.44, 107: 0.42, 108: 0.40, 109: 0.38,
            110: 0.36, 111: 0.35, 112: 0.33, 113: 0.31, 114: 0.29, 115: 0.27,
            116: 0.26, 117: 0.24, 118: 0.23, 119: 0.21, 120: 0.20, 121: 0.18,
            122: 0.17, 123: 0.15, 124: 0.14, 125: 0.13, 126: 0.11, 127: 0.10,
            128: 0.08, 129: 0.07, 130: 0.06, 131: 0.05, 132: 0.03, 133: 0.02,
            134: 0.01, 135: 0.00}


def get_tng_ages():
    # Values taken from tng100-1 webpage
    return {0: 0.179, 1: 0.271, 2: 0.37, 3: 0.418, 4: 0.475, 5: 0.517, 6: 0.547,
            7: 0.596, 8: 0.64, 9: 0.687, 10: 0.732, 11: 0.764, 12: 0.844, 13: 0.932,
            14: 0.965, 15: 1.036, 16: 1.112, 17: 1.177, 18: 1.282, 19: 1.366, 20: 1.466,
            21: 1.54, 22: 1.689, 23: 1.812, 24: 1.944, 25: 2.145, 26: 2.238, 27: 2.384,
            28: 2.539, 29: 2.685, 30: 2.839, 31: 2.981, 32: 3.129, 33: 3.285, 34: 3.447,
            35: 3.593, 36: 3.744, 37: 3.902, 38: 4.038, 39: 4.206, 40: 4.293, 41: 4.502,
            42: 4.657, 43: 4.816, 44: 4.98, 45: 5.115, 46: 5.289, 47: 5.431, 48: 5.577,
            49: 5.726, 50: 5.878, 51: 6.073, 52: 6.193, 53: 6.356, 54: 6.522, 55: 6.692,
            56: 6.822, 57: 6.998, 58: 7.132, 59: 7.314, 60: 7.453, 61: 7.642, 62: 7.786,
            63: 7.932, 64: 8.079, 65: 8.28, 66: 8.432, 67: 8.587, 68: 8.743, 69: 8.902,
            70: 9.062, 71: 9.225, 72: 9.389, 73: 9.556, 74: 9.724, 75: 9.837, 76: 10.009,
            77: 10.182, 78: 10.299, 79: 10.535, 80: 10.654, 81: 10.834, 82: 11.016,
            83: 11.138, 84: 11.323, 85: 11.509, 86: 11.635, 87: 11.824, 88: 11.951,
            89: 12.143, 90: 12.337, 91: 12.467, 92: 12.663, 93: 12.795, 94: 12.993,
            95: 13.127, 96: 13.328, 97: 13.463, 98: 13.667, 99: 13.803}


def get_illustris_ages():
    # Values taken from illustris-2 webpage
    # Note that snapshots 53 and 55 are missing for Illustris-1
    return {0: 0.054, 1: 0.058, 2: 0.062, 3: 0.066, 4: 0.071, 5: 0.076,
            6: 0.082, 7: 0.087, 8: 0.094, 9: 0.100, 10: 0.108, 11: 0.116,
            12: 0.123, 13: 0.132, 14: 0.142, 15: 0.153, 16: 0.163, 17: 0.175,
            18: 0.188, 19: 0.201, 20: 0.215, 21: 0.231, 22: 0.248, 23: 0.266,
            24: 0.283, 25: 0.304, 26: 0.327, 27: 0.351, 28: 0.376, 29: 0.401,
            30: 0.431, 31: 0.463, 32: 0.486, 33: 0.497, 34: 0.529, 35: 0.560,
            36: 0.568, 37: 0.610, 38: 0.655, 39: 0.703, 40: 0.750, 41: 0.782,
            42: 0.805, 43: 0.864, 44: 0.927, 45: 0.954, 46: 0.989, 47: 1.061,
            48: 1.139, 49: 1.205, 50: 1.223, 51: 1.312, 52: 1.399, 53: 1.501, 54: 1.577,
            55: 1.611, 56: 1.728, 57: 1.855, 58: 1.990, 59: 2.134, 60: 2.195, 61: 2.289,
            62: 2.438, 63: 2.596, 64: 2.745, 65: 2.902, 66: 3.047, 67: 3.198,
            68: 3.356, 69: 3.522, 70: 3.669, 71: 3.823, 72: 3.983, 73: 4.120,
            74: 4.291, 75: 4.438, 76: 4.590, 77: 4.747, 78: 4.908, 79: 5.074,
            80: 5.210, 81: 5.384, 82: 5.527, 83: 5.674, 84: 5.824, 85: 5.977,
            86: 6.015, 87: 6.172, 88: 6.292, 89: 6.455, 90: 6.622, 91: 6.791,
            92: 6.921, 93: 7.096, 94: 7.230, 95: 7.411, 96: 7.550, 97: 7.737,
            98: 7.880, 99: 8.024, 100: 8.171, 101: 8.369, 102: 8.520, 103: 8.672,
            104: 8.827, 105: 8.983, 106: 9.141, 107: 9.301, 108: 9.463, 109: 9.626,
            110: 9.791, 111: 9.902, 112: 10.070, 113: 10.240, 114: 10.411,
            115: 10.585, 116: 10.701, 117: 10.876, 118: 11.054, 119: 11.173,
            120: 11.353, 121: 11.534, 122: 11.656, 123: 11.839, 124: 11.963,
            125: 12.149, 126: 12.336, 127: 12.462, 128: 12.652, 129: 12.779,
            130: 12.971, 131: 13.100, 132: 13.294, 133: 13.424, 134: 13.620,
            135: 13.752}


def fill_histories_dataframe(config, arr):
    input_properties = ['bh_mass', 'bh_dot', 'dm_fof_mass', 'dm_sub_mass', 'gas_mass', 'sfr', 'stellar_mass',
                        'merge_bh_mass', 'merge_bh_dot', 'merge_dm_fof_mass', 'merge_dm_sub_mass',
                        'merge_gas_mass', 'merge_stellar_mass', 'n_minor_merge', 'n_major_merge']

    output_features = ['bh_mass', 'gas_mass', 'mock_g', 'mock_k', 'mock_r', 'mock_u',
                       'sfr', 'stellar_mass', 'stellar_metallicity',
                       'central', 'lowest_snap', 'subhalo_id', 'x', 'y', 'z']

    max_snap = config.snapshot_to_predict
    min_snap = config.get_min_snap_to_extract()
    snapshots = list(range(max_snap, min_snap-1, -1))
    n_input, n_output, n_snap = len(input_properties), len(output_features), len(snapshots)
    input_features = [str(snap)+prop for snap in snapshots for prop in input_properties]

    valid = (arr['snap_num'] == max_snap) & (arr['dm_sub_mass'] > config.get_mass_limit())
    n_valid_sub_this_file = np.sum(valid)
    histories = np.zeros((n_valid_sub_this_file, n_input*n_snap + n_output), dtype='float64')
    i_sub = 0
    n_halo = arr['snap_num'].shape[0]

    for i_halo in range(n_halo):
        # Checking if subhalo is valid
        if arr['snap_num'][i_halo] != max_snap:
            continue
        if arr['dm_sub_mass'][i_halo] < config.get_mass_limit():
            continue

        i_prog = i_halo
        snap_num = max_snap  # snap_num is immediately redefined, but this stops pycharm throwing an error
        while i_prog != -1:
            snap_num = arr['snap_num'][i_prog]
            if snap_num < min_snap:
                break

            bh_mass = arr['bh_mass'][i_prog]
            bh_dot = arr['bh_dot'][i_prog]
            dm_fof_mass = arr['dm_fof_mass'][i_prog]
            dm_sub_mass = arr['dm_sub_mass'][i_prog]
            gas_mass = arr['gas_mass'][i_prog]
            sfr = arr['sfr'][i_prog]
            stellar_mass = arr['stellar_mass'][i_prog]

            merge_bh_mass, merge_bh_dot, merge_dm_fof_mass, merge_dm_sub_mass = 0, 0, 0, 0
            merge_gas_mass, merge_stellar_mass, n_minor_merge, n_major_merge = 0, 0, 0, 0
            i_next = arr['next_prog_index'][i_prog]
            while i_next != -1:
                merge_bh_mass += arr['bh_mass'][i_next]
                merge_bh_dot += arr['bh_dot'][i_next]
                merge_dm_fof_mass += arr['dm_fof_mass'][i_next]
                merge_dm_sub_mass += arr['dm_sub_mass'][i_next]
                merge_gas_mass += arr['gas_mass'][i_next]
                merge_stellar_mass += arr['stellar_mass'][i_next]

                # Lifted from https://github.com/illustristng/illustris_python/blob/master/sublink.py#L185
                prog_stellar_mass = arr['stellar_mass'][i_prog]
                next_stellar_mass = arr['stellar_mass'][i_next]
                if (prog_stellar_mass > 0) and (next_stellar_mass > 0):
                    ratio = next_stellar_mass / prog_stellar_mass
                    if (ratio >= 1/3) and (ratio <= 3):
                        n_major_merge += 1
                    elif (ratio >= 1/10) and (ratio <= 10):
                        n_minor_merge += 1

                i_next = arr['next_prog_index'][i_next]

            i_start = (max_snap - snap_num) * n_input
            # This has to line up with where input columns are defined
            data = [bh_mass, bh_dot, dm_fof_mass, dm_sub_mass, gas_mass, sfr, stellar_mass,
                    merge_bh_mass, merge_bh_dot, merge_dm_fof_mass, merge_dm_sub_mass,
                    merge_gas_mass, merge_stellar_mass, n_minor_merge, n_major_merge]
            histories[i_sub, i_start:i_start+n_input] = data

            i_prog = arr['main_prog_index'][i_prog]

        bh_mass = arr['bh_mass'][i_halo]
        gas_mass = arr['gas_mass'][i_halo]
        mock_g = arr['mock_g'][i_halo]
        mock_k = arr['mock_k'][i_halo]
        mock_r = arr['mock_r'][i_halo]
        mock_u = arr['mock_u'][i_halo]
        sfr = arr['sfr'][i_halo]
        stellar_mass = arr['stellar_mass'][i_halo]
        stellar_metallicity = arr['stellar_metallicity'][i_halo]

        central = arr['is_central'][i_halo]
        lowest_snap = max(snap_num, min_snap)
        subhalo_id = arr['subhalo_id'][i_halo]
        x, y, z = arr['x'][i_halo], arr['y'][i_halo], arr['z'][i_halo]

        # This has to line up with where output columns are defined
        data = [bh_mass, gas_mass, mock_g, mock_k, mock_r, mock_u,
                sfr, stellar_mass, stellar_metallicity,
                central, lowest_snap, subhalo_id, x, y, z]
        histories[i_sub, n_input*n_snap:] = data

        i_sub += 1

    return pd.DataFrame(histories, columns=input_features+output_features)
