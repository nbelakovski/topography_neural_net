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
import copy
from time import time
from pgmagick import Image
from multiprocessing import Process
import http.server

def launch_server():
	sys.stdout = open(str(os.getpid())+'.out','w')
	sys.stderr = open(str(os.getpid())+'.err','w')
	httpd = http.server.HTTPServer(('', 8000), http.server.SimpleHTTPRequestHandler)
	httpd.serve_forever()

p = Process(target=launch_server)
p.start()

def create_plot(original_data, interpolated_data):
	plot_data = [go.Surface(z=original_data), go.Surface(z=interpolated_data)]

	# Set up the camera so that the orientation of the surface is similar to the orientation of the associated image
	camera = dict(
	        up=dict(x=1, y=0, z=0),
	        center=dict(x=0, y=0, z=0),
	        eye=dict(x=.6, y=-.2, z=1.5)
	)
	camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=-1.4, z=1.5)
	)

	# Set up subplots. One for the lidar image, next for the original data, next for the machine learned data
	# The first subplot isn't actually used - the image is carefully placed to take up that slot
	fig = pt.make_subplots(1, 2, specs=[[{'is_3d': True}, {'is_3d': True}]])
	fig.append_trace(plot_data[0], 1, 1)
	fig.append_trace(plot_data[1], 1, 2)
	# fig['layout'].update(layout)
	fig['layout'].update(scene1=dict(camera=camera), scene2=dict(camera=camera))

	a = po.plot(fig)
	print("Done")

data_dir = sys.argv[1]
from glob import glob
data_dirs = os.listdir(data_dir)
for d in data_dirs:
	# Check to see if this directory has been reviewed
	if os.path.exists(os.path.join(data_dir, d, 'reviewed')):
		continue
	# If not, create the jpg file for hosting
	jp2_file = Image(os.path.join(data_dir, d, 'cropped.jp2'))
	jpg_file = jp2_file.write('cropped.jpg')
	# Now load the data file
	data_filename = glob(os.path.join(data_dir, d, '*data'))[0]
	z_data = read_data(data_filename)
	orig_data = copy.deepcopy(z_data)
	time1 = time()
	rcode = interpolate_zeros_2(z_data)
	time2 = time()
	print("Interpolation took", (time2 - time1), "seconds")

	# Now create the plotly page
	create_plot(orig_data, z_data)
	a = input("Keep data for directory " + d + "?([y]/n)")
	print(a)
	if a == 'y' or a == '':
		with open(os.path.join(data_dir, d, 'reviewed'), 'w') as f:
			f.write('')
	else:
		with open(os.path.join(data_dir, d, 'marked'), 'w') as f:
			f.write('')
	
	# Then need to do the raw input thing, setup deletion and marking of review

a = input("Test")
p.terminate()
	


