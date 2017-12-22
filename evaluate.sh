#!/bin/bash -x

IMAGE=$1
DATA=$2

# This will create a file called test.pickle, which will be autoimported by visualization.py
python3 inference.py $IMAGE
python3 visualization.py $DATA
mv temp-plot.html docs/temp-plot.html
python3 -c "from pgmagick import Image; Image('$IMAGE').write('docs/cropped.jpg')"