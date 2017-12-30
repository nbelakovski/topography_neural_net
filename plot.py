#!/usr/bin/python3

import plotly.offline as po
import plotly.tools as pt
import plotly.graph_objs as go
import sys

batch_evals = []
epoch_evals = [(0,0)] # Add a point at the origin just to make the graph look nice
data_file = open(sys.argv[1], 'r')
gather_batch_data = False
for line in data_file:
	# We want to only gather data for eval during the training,
	# not the evals at the end. To do so, we enter "gather mode"
	# between the instance of "Running epoch #" and "Epoch # completed"
	if "Running epoch" in line:
		batch_evals_for_epoch = []
		gather_batch_data = True
		continue
	elif "completed" in line:
		gather_batch_data = False
		# Some data processing. Want to normalize the intervals so that things line up on the graph
		epoch = int(line.split(" ")[1])
		batch_evals_for_epoch = [(x[0]/len(batch_evals_for_epoch) + epoch, x[1]) for x in batch_evals_for_epoch]
		batch_evals.extend(batch_evals_for_epoch)
		continue
	# Now, if we're in gather mode, get the relevant data and put it in batch_evals
	if gather_batch_data and 'eval' in line:
		pieces = line.split(' ')
		accuracy = float(pieces[-1])
		batch_evals_for_epoch.append((len(batch_evals_for_epoch) + len(epoch_evals), accuracy))
	# While outside of batch mode, we grab the epoch level evaluation
	elif "R2" in line:
		epoch_accuracy = float(line.split(":")[1])
		epoch = int(line.split(" ")[3]) + 1
		epoch_evals.append((epoch, epoch_accuracy))

# Strip out all training after the last epoch
batch_evals = [x for x in batch_evals if x[0] < epoch_evals[-1][0]]
# Subsample the data so there's a reasonable number of points in between epochs
number_of_points_per_epoch = 30
print(batch_evals[:10])
batch_evals = [x for i,x in enumerate(batch_evals) if i % int(len(batch_evals)/len(epoch_evals) / number_of_points_per_epoch) == 0]
print(batch_evals[:10])

plot_data = [go.Scatter(x=[x[0] for x in batch_evals], y=[x[1] for x in batch_evals], name='partial eval'),
			 go.Scatter(x=[x[0] for x in epoch_evals], y=[x[1] for x in epoch_evals], name='full eval')]

layout = go.Layout(
	title='Coefficient of Determination (R<sup>2</sup>) vs Epoch',
    xaxis=go.XAxis(
        range=[0, len(epoch_evals) - 0.5],
        title='<br>Epoch', # Need the <br> in the front to prevent the axis title from clashing with the plot title
        dtick=1,
        side="top"
    ),
    yaxis=go.YAxis(
        title='Coefficient of Determination (R<sup>2</sup>)',
        range=[-1600, 1]
    ),
)

fig = go.Figure(data=plot_data, layout=layout)

# PLOT!
a = po.plot(fig, show_link=False, output_type='div')
with open('docs/training_statistics.html', 'w') as f:
	f.write(a)
print("Done")
