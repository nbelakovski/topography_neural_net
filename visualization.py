# Goal of this file is to take an image and a topography dataset and make some 3D representation, or 2D with colormap,
# or something
import pickle

import os
import plotly.offline as po
import plotly.tools as pt
import plotly.graph_objs as go
import json

import laspy

with open('data/paired_data.json', 'r') as f:
    data = json.load(f)
# Read data from LAS file
# Eventually will need to associate an image with its metadata and do cropping, but one thing at a time, let's focus on
# figuring out the plotly interface for the topographical data
for i, scene in enumerate(data):
    las_filename = 'data/' + str(i) + '/' + scene[0]['displayId'] + '/' + scene[0]['displayId'] + '.las'
    print(las_filename)
    if os.path.isfile(las_filename) == False:
        continue
    f = laspy.file.File(las_filename)
    scale = 2000
    z_data = [f.Z[i*scale] for i in range(0, int(len(f.Z) / scale))]
    plot_data = [
        go.Scatter3d(
                x = [f.X[i*scale] for i in range(0, int(len(f.X) / scale))],
                y = [f.Y[i*scale] for i in range(0, int(len(f.Y) / scale))],
                z = z_data,
                mode = 'markers',
                marker = {"showscale": True, "color": z_data}
        )
    ]

    camera = dict(
            up=dict(x=1, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-2, z=1)
    )
    layout = go.Layout(
        title='Elevation near Glacier Peak, WA',
        autosize=True,
        # width=1200,
        # height=800,
        # margin=dict(
        #         l=65,
        #         r=50,
        #         b=65,
        #         t=90
        # ),
        images=[dict(
                source='https://raw.githubusercontent.com/nbelakovski/topography_neural_net/master/WA_GlacierPeak_2014_000070.jpg',#'https://raw.githubusercontent.com/cldougl/plot_images/add_r_img/vox.png', #'https://earthexplorer.usgs.gov/browse/lidar/WA/2014/WA_GlacierPeak_2014/WA_GlacierPeak_2014_000070.jpg', #scene[0]['browseUrl'],
                xref="paper", yref="paper",
                x=-0.07, y=0.83,
                sizing="stretch",
                sizex=0.4, sizey=0.7, layer="below",
                xanchor = 'left', yanchor='top'
        )],
    )
    print(scene[0]['browseUrl'])

    # fig = go.Figure(data=plot_data, layout=layout)
    fig = pt.make_subplots(1,3, specs=[[{'is_3d':True}, {'is_3d':True}, {'is_3d':True}]])
    fig.append_trace(plot_data[0], 1, 2)
    fig['layout'].update(layout)
    fig['layout'].update(scene2=dict(camera=camera))

    a = po.plot(fig)
    print(a)

pass