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
from data.utils import interpolate_zeros_2
from tools.tools import read_data
import utils

data_filename = sys.argv[1]
# Load the matrix containing the original topographical data
z_data = read_data(data_filename)
interpolate_zeros_2(z_data)
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

layout = go.Layout(
    title='[original topography, estimated topography]',
    autosize=True,
)

# Set up subplots. One for the lidar image, next for the original data, next for the machine learned data
# The first subplot isn't actually used - the image is carefully placed to take up that slot
fig = pt.make_subplots(1, 2, specs=[[{'is_3d': True}, {'is_3d': True}]])
fig.append_trace(plot_data[0], 1, 1)
fig.append_trace(plot_data[1], 1, 2)
fig['layout'].update(layout)
fig['layout'].update(scene1=dict(camera=camera), scene2=dict(camera=camera))

# PLOT!
a = po.plot(fig, show_link=False, filename="docs/plots.html")
print("Done")
