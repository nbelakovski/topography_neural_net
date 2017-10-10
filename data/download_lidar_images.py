import json
import os
import subprocess

# This file is meant to download the colorscale images associated with the lidar data

paired_data = json.load(open('paired_data.json', 'r'))

for i in range(len(paired_data)):
    # noinspection PyArgumentList
    os.chdir(str(i))
    url = paired_data[i][0]['browseUrl']
    subprocess.call(['wget', '--progress=bar:force', '--trust-server-names', '--content-disposition', url])
    os.chdir('..')
    if i > 110:
        break
