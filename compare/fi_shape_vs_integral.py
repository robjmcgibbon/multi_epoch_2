import datetime
import os
import sys

import bokeh.models.tools as bokeh_tools
import bokeh.models.ranges as bokeh_ranges
import bokeh.plotting as bokeh
import matplotlib.pyplot as plt

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

log(f'Running {__file__}')
main_config = helpers.Config('config.yaml')
secondary_configs = main_config.load_secondary_configs('config.yaml')
configs = [main_config] + secondary_configs
config_names = [config.name for config in configs]
config_snapshots = [
    config.get_standard_spacing()
    # config.get_standard_spacing_one_snapshot_early()
    # config.get_tight_spacing()
    # config.get_every_snapshot()
    for config in configs
]
datasets = [config.load_data(snapshots) for config, snapshots in zip(configs, config_snapshots)]

# TODO: For comparing very large number of simulations, this plot could be a heatmap

output_features = {
    'gas_mass': ['bh_mass', 'dm_sub_mass', 'stellar_mass'],
    'sfr': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
    'stellar_mass': ['bh_mass', 'dm_sub_mass', 'gas_mass'],
    'stellar_metallicity': ['bh_mass', 'dm_sub_mass', 'gas_mass', 'stellar_mass'],
}

shape_measure = 'half'         # Can be 'half' or 'peak'
plot_lib = 'matplotlib'        # Can be 'matplotlib' or 'bokeh'
m_marker_shapes = ['o', 'v', 's', 'D'] * 5

for i_output, (output_feature, input_properties) in enumerate(output_features.items()):

    # Defining shared plot information
    plot_name = f'btml_{output_feature}_shape_vs_integral'
    plot_title = f'Predicting {main_config.get_proper_name(output_feature, False)}'
    x_label = 'Total feature importance over all snapshots'
    if shape_measure == 'half':
        y_label = 'Universe age where half of total feature importance is reached [Gyr]'
    elif shape_measure == 'peak':
        y_label = 'Universe age where feature importance peaks [Gyr]'
    else:
        raise AssertionError

    # Setting up matplotlib plot
    m_fig, ax = plt.subplots(1)
    ax.set_xlim(0)
    ax.set_title(f'Predicting {main_config.get_proper_name(output_feature, False)}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Setting up bokeh plot
    # TODO: Add images to hoover tips
    # https://docs.bokeh.org/en/latest/docs/user_guide/tools.html#custom-tooltip
    bokeh.output_file(filename=main_config.plot_dir+plot_name+'.html', title='Shape vs Integral')
    b_x_range = bokeh_ranges.Range1d(start=-0.01, end=0)
    b_fig = bokeh.figure(
        title=plot_title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_range=b_x_range,
        plot_width=1000,
        plot_height=800,
        tools=[bokeh_tools.HoverTool()],
        #         tooltips='$name',
        tooltips=r'<div style="font-size:x-large;">$name</div>'
    )
    b_fig.title.align = 'center'
    b_fig.title.text_font_size = '25pt'
    b_fig.axis.axis_label_text_font_size = '20pt'
    b_fig.axis.axis_label_text_font_style = 'normal'
    b_fig.axis.major_label_text_font_size = '15pt'

    b_marker_size = 20
    b_marker_shapes = [
            b_fig.circle,
            b_fig.diamond,
            b_fig.hex,
            b_fig.inverted_triangle,
            b_fig.plus,
            b_fig.square,
            b_fig.square_pin,
            b_fig.star,
            b_fig.triangle,
            b_fig.triangle_pin,
        ]
    b_colors = ['black'] + [main_config.get_color(prop) for prop in input_properties]
    b_prop_labels = ['dummy'] + [main_config.get_proper_name(prop, False) for prop in input_properties]

    # Calculating values to plot
    for i_config, (config, data, snapshots) in enumerate(zip(configs, datasets, config_snapshots)):

        ages = config.get_ages()

        # Setting up bokeh plotting arrays
        b_x_values = [-1]
        b_y_values = [ages[config.snapshot_to_predict]]

        input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
        mean_importance, _ = config.calculate_feature_importance(data, input_features, output_feature)

        mean_importance = {feat: val for feat, val in zip(input_features, mean_importance)}
        for input_property in input_properties:
            # TODO: Do a proper integral, not just a sum
            integral = 0
            shape_snap = snapshots[0]
            for snap in snapshots:
                integral += mean_importance[str(snap)+input_property]
                if mean_importance[str(snap)+input_property] > mean_importance[str(shape_snap)+input_property]:
                    shape_snap = snap

            if shape_measure == 'half':
                half_integral = 0
                for snap in snapshots:
                    half_integral += mean_importance[str(snap)+input_property]
                    if half_integral > 0.5 * integral:
                        shape_snap = snap
                        break
            shape_age = ages[shape_snap]

            # Plotting matplotlib points
            ax.plot(integral, shape_age,
                    m_marker_shapes[i_config],
                    markersize=10,
                    color=config.get_color(input_property))

            # Adding data to bokeh plotting arrays
            b_x_values.append(integral)
            b_x_range.end = max(b_x_range.end, integral*1.1)
            b_y_values.append(shape_age)

        # Plotting bokeh points
        b_marker_shapes[i_config](b_x_values, b_y_values,
                                  size=b_marker_size,
                                  fill_color=b_colors,
                                  line_width=0,
                                  name=config.name,
                                  legend_label=config.name,
                                  )

    # TODO: Add redshift twin axis for both plots

    # Adding legends to maplotlib plot. Saving plot
    config_markers = []
    for i_config, config in enumerate(configs):
        p = ax.plot(-1, 7, m_marker_shapes[i_config], markersize=10, color='k')
        config_markers.append(p[0])
    config_legend = ax.legend(config_markers, config_names, loc='upper left', fontsize='small', framealpha=0.5)
    ax.add_artist(config_legend)

    property_markers, property_labels = [], []
    for i_prop, prop in enumerate(input_properties):
        p = ax.plot(-1, 7, 'o', markersize=10, color=main_config.get_color(prop))
        property_markers.append(p[0])
        property_labels.append(main_config.get_proper_name(prop, False))
    property_legend = ax.legend(property_markers, property_labels, loc='lower right', fontsize='small', framealpha=0.5)
    ax.add_artist(property_legend)

    main_config.plot_show_save(plot_name, fig, force_save=True)

    # Adding legend giving property colors to bokeh plot. Saving plot
    for i in range(1, len(b_colors)):
        b_fig.circle(-1, -1,
                     size=b_marker_size,
                     line_width=0,
                     fill_color=b_colors[i],
                     legend_label=b_prop_labels[i],
                     )

    b_fig.legend.click_policy = 'hide'
    b_fig.legend.label_text_font_size = '16pt'
    b_fig.legend.glyph_height = 50
    b_fig.legend.glyph_width = 50
    b_fig.add_layout(b_fig.legend[0], 'right')

    bokeh.save(b_fig)

log(f'Finished {__file__} with runtime of {datetime.datetime.now()-main_config.start_time}\n')
