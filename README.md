# topography_neural_net
Attempting to estimate topography of a region from image data

## Introduction
The ultimate aim of this project is to create a neural net that can input satellite image data and output a matrix repesenting
the topology of the area imaged. It will not be a pixel-level mapping, but close. The reason to avoid pixel-level mapping is both
to keep the network size down for training purposes, and also because I am more interested in capturing the "broad strokes" of a
region's topography as opposed to fine details.

## Motivation
The primary motivation for this project is as a learning exercise for myself in machine learning and convolutional neural nets.
This project gives me an opportunity to establish a data pipeline, design a neural net, implement it, and train it and examine
the results. I've definitely learned a lot from this opportunity to get my hands dirty, chiefly that the largest challenge is not
in the network architecture, but the data gathering. The network architecture is not trivial, but once a network is established
it is fairly easy to tweak it. Getting the data pipeline built required answering many more abstract questions and required a
lot of work to set up and scale in a way that would get me meaningful amount of data for training.

## Potential applications
I must admit to a sincere ignorance of current methods of gathering topography data, and their various advantages and disadvantages.
The list below is my best guess at situations and applications in which this model might have some usefulness. I welcome and
appreciate any additions or clarifications anyone would like to make.
* Emergency services - In the event of a major landslide, it would be very useful to know the new topography of a location.
                     I presume that instruments which accurately topography might take some time to deploy, whereas there are
                     several companies launching satellite fleets that will attempt to image all of the Earth once every 24h.
                     So, if this net works even to some reasonable accuracy it could be useful in these emergency scenarios.
* Planetary landing - For probes that land on planetary bodies, it is crucial to know the topography in order to ensure landing
                    on a relatively flat and debris-free surface. Images obtained during the descent process could be transformed
                    into topography with this net and this information could be fed into the landing computer, ensuring a safe
                    and level landing.
* Hard-to-reach areas - Perhaps there are areas whose topology is difficult to estimate accurately due to their, ahem, topology.
                      A model such as this one could provide some first order estimates.
                      
## Example
[Here](https://nbelakovski.github.io/topography_neural_net/index.html) you will find an example of the end goal. On the left is a satellite image. In the middle is the original
topography data as taken from a LIDAR scan, and on the right will be the output of the neural net when it is applied to the image
on the left (once the network is sufficiently trained). The image on the left will not be in the training set.

## Data source
The data is coming entirely from the United States Geological Survey (USGS). The image data is from the National Agriculture
Imagery Program (NAIP) and the topography data is LIDAR data that is hosted by the USGS, but comes from "contracts, partnerships
with other Federal, state, tribal, or regional agencies, from direct purchases from private industry vendors, and through
volunteer contributions from the science community."

## Network design
More detail will be provided shortly. The basic design is a convolutional neural net with 4 convolution-pooling layers, followed
by a fully connected layer of ~1000 neurons (final numbers subject to change), followed by the output layer which is a 250x250
matrix of height points. The input image is ~1000x~1000 pixels and is in color. It might be changed to grayscale in order to
accomodate larger layers downstream of the input. The network is trained with gradient descent using the Adam optimizer and a
batch size of 3. In the current, non-final configuration, the network has over 75 million parameters, of which 62.5 million are
in the layer connecting the 1000 neurons to the final 250x250 matrix, and 12.5 million are in the layer connecting the output of
the final pooling layer to the 1000 neuron penultimate layer. The network as currently designed uses nearly all of the available
8GB of memory in the GPU on the training machine

## Training
The current data set consists of 91 pairs of [image, LAS file] (the LAS file contains the LIDAR data). This will be expanded
to ~1000 pairs or more once the data pipeline is streamlined to take up less disk space. Each pair currently takes up nearly 1 GB
of disk space, but this will be optimized by throwing out extraneous data after preprocessing (as you might expect, the LIDAR
data takes up the lion's share of the space, but most of it is as a higher resolution than what I believe is necessary for this
project).
Training takes place on a machine rented from Paperspace with a nVidia Quadro M4000.

## Final training statistics
TODO!
