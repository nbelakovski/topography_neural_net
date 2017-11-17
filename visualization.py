# Goal of this file is to take an image and a topography dataset and make some 3D representation
import pickle
import os
import plotly.offline as po
import skimage.measure
import plotly.tools as pt
import plotly.graph_objs as go
import numpy
import numpy as np
from data.subsample_matrix import subsample_matrix
import sys
from data.utils import read_data, interpolate_zeros
import utils

data_filename = sys.argv[1]
# Load the matrix containing the original topographical data
z_data = read_data(data_filename)
interpolate_zeros(z_data)
new_shape = utils.evenly_divisible_shape(z_data.shape, 16)
z_data = z_data[0:new_shape[0], 0:new_shape[1]]
# Pool it down to the same size as the output of the net
m = skimage.measure.block_reduce(z_data, block_size=(16, 16), func=np.mean) # pool down
m = m.astype(float)
m -= m.mean()

# Import the result of inference on the cropped.jp2. This file is created by inference.py
test = pickle.load(open('test.pickle', 'rb'))
test /= 10000
test -= 1
test *= 2
test *= 600000
test *= -1
test = numpy.flipud(test)
# Create a surface plot
plot_data = [go.Surface(z=m), go.Surface(z=test)]

# Set up the camera so that the orientation of the surface is similar to the orientation of the associated image
camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-.4, z=2.25)
)

# Grab the image associated with the lidar dataset to display next to the 3d surface plot. For now, images will be
# stored in my github
image_source = 'https://raw.githubusercontent.com/nbelakovski/topography_neural_net/master/sample_data/01430d/cropped.jpg'
layout = go.Layout(
    title='Estimating Topography from Image Data (Work in Progress)' + '<br><br>' + '[original image, original topography, estimated topography]',
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
    annotations=[
        dict(
            x=.975,
            y=.1,
            xref='paper',
            yref='paper',
            text='WORK IN PROGRESS', #'TODO:' + '<br>' + 'Neural Net Topography' + '<br>' + 'Output Here',
            showarrow=False,
            font=dict(
                    size=18,
            )
        )
    ],
)

# Set up subplots. One for the lidar image, next for the original data, next for the machine learned data
# The first subplot isn't actually used - the image is carefully placed to take up that slot
fig = pt.make_subplots(1, 3, specs=[[{'is_3d': False}, {'is_3d': True}, {'is_3d': True}]])
fig.append_trace(plot_data[0], 1, 2)
fig.append_trace(plot_data[1], 1, 3)
fig['layout'].update(layout)
fig['layout'].update(scene1=dict(camera=camera), scene2=dict(camera=camera))

# PLOT!
a = po.plot(fig)
print("Done")
