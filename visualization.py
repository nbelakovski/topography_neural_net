# Goal of this file is to take an image and a topography dataset and make some 3D representation

import os
import plotly.offline as po
import plotly.tools as pt
import plotly.graph_objs as go
import json
import pickle
import numpy


with open('data/paired_data.json', 'r') as f:
    data = json.load(f)

for i, scene in enumerate(data):
    # Load the topographical information in matrix form. The pickle file should have been created by
    # convert_las_to_matrix.py
    pickle_filename = 'data/' + str(i) + '/' + scene[0]['displayId'] + '.pickle'
    if os.path.isfile(pickle_filename) is False:
        continue
    z_data = pickle.load(open(pickle_filename, 'rb'))

    # scale matrix down to something that can be reasonably loaded in an html page
    matrix_size = 300
    m = numpy.zeros([matrix_size, matrix_size])
    scale_factor = int(z_data.shape[0] / matrix_size)
    for row in range(0, m.shape[0]):
        zi = int(row/m.shape[0] * (z_data.shape[0] - 1))
        for col in range(0, m.shape[1]):
            zj = int(col/m.shape[1] * (z_data.shape[1] - 1))
            m[row, col] = z_data[zi, zj]

    # Create a surface plot
    plot_data = [go.Surface(z=m)]

    # Set up the camera so that the orientation of the surface is similar to the orientation of the associated image
    camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-1.6, z=1.35)
    )

    # Grab the image associated with the lidar dataset to display next to the 3d surface plot. For now, images will be
    # stored in my github
    image_filename = scene[0]['browseUrl'].split('/')[-1]
    image_source = 'https://raw.githubusercontent.com/nbelakovski/topography_neural_net/master/data/' + str(i) + '/' + \
                   image_filename
    layout = go.Layout(
        title='Elevations near Glacier Peak, WA',
        autosize=True,
        images=
        [
            dict
            (
                source=image_source,
                xref="paper", yref="paper",
                x=-0.07, y=0.83,
                sizing="stretch",
                sizex=0.4, sizey=0.7, layer="below",
                xanchor='left', yanchor='top'
            )
        ],
    )

    # Set up subplots. One for the lidar image, next for the original data, next for the machine learned data
    # The first subplot isn't actually used - the image is carefully placed to take up that slot
    fig = pt.make_subplots(1, 3, specs=[[{'is_3d': False}, {'is_3d': True}, {'is_3d': True}]])
    fig.append_trace(plot_data[0], 1, 2)
    fig['layout'].update(layout)
    fig['layout'].update(scene1=dict(camera=camera))  # need to look at console output to determine which key to update

    # PLOT!
    a = po.plot(fig)
