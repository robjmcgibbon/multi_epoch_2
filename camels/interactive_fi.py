# Use the ``bokeh serve`` command to run the example by executing:
# $ bokeh serve interactive_fi.py
# Then navigate to the URL:
# http://localhost:5006/interactive_fi

import os
import sys

import joblib
import numpy as np
import yaml

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Slider
from bokeh.plotting import figure

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers


# TODO: Side by side plots - Illustris vs SIMBA
sim = 'IllustrisTNG'
# TODO: Subplots with different output features
output_feature = 'stellar_mass'

interpolate_dir = f'{helpers.Config.get_base_dir()}generated/baryon_tree_ml/camels/interpolate/{sim}/'
interpolator_name = f'{output_feature}_interpolator.joblib'
interpolator = joblib.load(interpolate_dir+interpolator_name)
with open(interpolate_dir+'output_features.yaml', 'r') as yaml_file:
    input_properties = yaml.safe_load(yaml_file)[output_feature]
snapshots = np.load(interpolate_dir+'snapshots.npy')
input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]


def generate_data(params):
    data = {'snapshots': snapshots}
    mean_importance = interpolator.predict([params])[0]
    for input_property in input_properties:
        mean_values = []
        for snap in snapshots:
            idx = input_features.index(str(snap)+input_property)
            mean_values.append(mean_importance[idx])
        data[f'{input_property}'] = np.array(mean_values)
    return data, np.max(mean_importance)


initial_data, initial_y_max = generate_data([0.3, 0.8, 0, 0, 0, 0])
source = ColumnDataSource(data=initial_data)

# TODO: Plot redshift on x axis
padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
plot = figure(height=700, width=900,
              title=f'Predicting z=0 {helpers.Config.get_proper_name(output_feature, False)}',
              tools="crosshair,save",
              x_range=[np.min(snapshots)-padding, np.max(snapshots)+padding],
              y_range=[0, initial_y_max*1.2],
              x_axis_label='Snapshot',
              y_axis_label='Feature Importance')
plot.title.text_font_size= '20pt'
plot.title.align = 'center'
plot.xaxis.axis_label_text_font_size = '14pt'
plot.xaxis.axis_label_text_font_style = 'bold'
plot.yaxis.axis_label_text_font_size = '14pt'
plot.yaxis.axis_label_text_font_style = 'bold'
plot.toolbar.autohide = True

for input_property in input_properties:
    plot.line('snapshots', input_property, source=source, line_width=3,
              color=helpers.Config.get_color(input_property),
              legend_label=helpers.Config.get_proper_name(input_property, False))

plot.legend.location = 'top_left'
plot.legend.click_policy = 'hide'
plot.legend.label_text_font_size = '14pt'

# TODO: Similar plot, except change galaxy subsample
# TODO: Change titles to be more physical, e.g. supernova wind
slider_title = Div(text='<b>CAMELS hyperparameters</b>', style={'font-size': '16pt'})
slider_desc = 'Adjust the sliders to see the effect on the feature importance.' \
              'If the values get too big the axes will adjust.'
slider_desc = Div(text=slider_desc)
omega_m = Slider(title='Omega matter', value=0.3, start=0.1, end=0.5, step=0.05)
sigma_8 = Slider(title='Sigma 8', value=0.8, start=0.6, end=1, step=0.05)
a_sn_1 = Slider(title='Supernova 1', value=0, start=-2, end=2, step=0.5)
a_agn_1 = Slider(title='AGN 1', value=0, start=-2, end=2, step=0.5)
a_sn_2 = Slider(title='Supernova 2', value=0, start=-1, end=1, step=0.25)
a_agn_2 = Slider(title='AGN 2', value=0, start=-1, end=1, step=0.25)


# Set up callbacks, lifted from bokeh example
def update_data(attrname, old, new):
    params = [
        omega_m.value,
        sigma_8.value,
        a_sn_1.value,
        a_agn_1.value,
        a_sn_2.value,
        a_agn_2.value
    ]
    data, y_max = generate_data(params)
    source.data = data
    if (y_max > 0.95 * plot.y_range.end) or (y_max < 0.6 * plot.y_range.end):
        plot.y_range.end = y_max * 1.2


for w in [omega_m, sigma_8, a_sn_1, a_agn_1, a_sn_2, a_agn_2]:
    w.on_change('value', update_data)

inputs = column(slider_title, slider_desc, Div(text=' '),
                omega_m, sigma_8, a_sn_1, a_agn_1, a_sn_2, a_agn_2)

curdoc().add_root(row(plot, Div(text=' '), inputs, width=1200))
curdoc().title = 'Interactive Feature Importance'
