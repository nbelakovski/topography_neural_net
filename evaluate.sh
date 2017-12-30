#!/bin/bash -x

IMAGE=$1
DATA=$2

# This will create a file called test.pickle, which will be autoimported by visualization.py
python3 inference.py $IMAGE

python3 visualization.py $DATA

# Convert the image from jp2 to jpg and place it in the right folder
python3 -c "from pgmagick import Image; Image('$IMAGE').write('docs/image.jpg')"