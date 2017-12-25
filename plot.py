#!/usr/local/bin/python3

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
	# between the instance of "Running rpoch #" and "Epoch # completed"
	if "Running" in line:
		gather_batch_data = True
		continue
	elif "completed" in line:
		gather_batch_data = False
		continue
	# Now, if we're in gather mode, get the relevant data and put it in batch_evals
	if gather_batch_data and 'eval' in line:
		pieces = line.split(' ')
		accuracy = float(pieces[-1])
		batch_evals.append((len(batch_evals) + len(epoch_evals), accuracy))
	# While outside of batch mode, we grab the epoch level evaluation
	elif "R2" in line:
		epoch_accuracy = float(line.split(":")[1])
		epoch_evals.append((len(batch_evals), epoch_accuracy))

# Strip out all training after the last epoch
batch_evals = [x for x in batch_evals if x[0] < epoch_evals[-1][0]]
# Subsample the data so there's ~20 points in between epochs
number_of_points_between_epochs = 33
batch_evals = [x for x in batch_evals if x[0] % int(epoch_evals[1][0]/number_of_points_between_epochs) == 0]
print(batch_evals)

plot_data = [go.Scatter(x=[x[0]/epoch_evals[1][0] for x in batch_evals], y=[x[1] for x in batch_evals], name='partial eval'),
			 go.Scatter(x=[x[0]/epoch_evals[1][0] for x in epoch_evals], y=[x[1] for x in epoch_evals], name='full eval')]

layout = go.Layout(
	title='Coefficient of Determination (R<sup>2</sup>) vs Epoch',
    xaxis=go.XAxis(
        range=[0, len(epoch_evals) - 1],
        title='Epoch',
        dtick=1
    ),
    yaxis=go.YAxis(
    	# trying to make this range semi-automatic. Take the min of either ther average of the batch evals or the
    	# smallest (i.e. worst) epoch eval. Then multiply that by 1.1 to have a little extra room
        range=[min(sum([x[1] for x in batch_evals])/len(batch_evals), min([x[1] for x in epoch_evals])) * 1.1, 1],
        title='Coefficient of Determination (R<sup>2</sup>)'
    ),
)

fig = go.Figure(data=plot_data, layout=layout)

# PLOT!
a = po.plot(fig, filename='training_statistics.html', show_link=False, output_type='div')
with open('a.html', 'w') as f:
	f.write(a)
print("Done")
