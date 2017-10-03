# Goal of this file is to take an image and a topography dataset and make some 3D representation, or 2D with colormap,
# or something

import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go

import laspy


# Read data from LAS file
# Eventually will need to associate an image with its metadata and do cropping, but one thing at a time, let's focus on
# figuring out the plotly interface for the topographical data
f = laspy.file.File('WA_GlacierPeak_2014_000167/WA_GlacierPeak_2014_000167.las')
scale = 2000
z_data = [f.Z[i*scale] for i in range(0, int(len(f.Z) / scale))]
data = [
    go.Scatter3d(
            x = [f.X[i*scale] for i in range(0, int(len(f.X) / scale))],
            y = [f.Y[i*scale] for i in range(0, int(len(f.Y) / scale))],
            z = z_data,
            mode = 'markers',
            marker = {"showscale": True, "color": z_data}
    )
]

layout = go.Layout(
    title='Elevation near Glacier Peak, WA',
    autosize=True
)

fig = go.Figure(data=data, layout=layout)
a = po.plot(fig)
print(a)

pass