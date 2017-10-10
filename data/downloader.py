import json
import os
import requests
import sys
import subprocess
from time import sleep

paired_data = json.load(open('paired_data.json', 'r'))

API_key = "39fa86f1e5454638a9d4bfc78f5e2037"

# Make directories for each pair of data, and put the information for each pair into each folder
for i, pair_data in enumerate(paired_data):
    # noinspection PyArgumentList
    os.makedirs(name=str(i), exist_ok=True)
    json_filename = str(i) + '/' + str(i) + '.json'
    json.dump(pair_data, open(json_filename, 'w'), indent=4)

del paired_data  # clean up


# Now query USGS for download info
# Get the API key
def get_api_key():
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/login'
    password = open('../password.txt', 'r').readline()
    data = {"username": "nbelakovski", "password": password, "authType": "EROS", "catalogId": "EE"}
    r = requests.post(login_url, data={'jsonRequest': json.dumps(data)})
    if r.json()['errorCode'] is None:
        global API_key
        API_key = r.json()['data']
        print("Obtained API key")
    else:
        print("Failure to obtain API key, errorCode: ", r.json()['errorCode'])
        sys.exit(-1)


# find all wget process and block from returning until they're all completed
def find_number_of_wget_processes():
    ps = subprocess.Popen("ps -ef | grep wget | wc -l", shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    ps.stdout.close()
    ps.wait()
    return int(output)


def download_data(dataset, entityid: int):
    rcode = None
    downloadoptions_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/downloadOptions'
    data = {"apiKey": API_key, "datasetName": dataset, "entityIds": entityid}
    r = requests.post(downloadoptions_url, data={'jsonRequest': json.dumps(data)})
    if r.json()['errorCode'] is None:
        for download_option in r.json()['data'][0]['downloadOptions']:
            if download_option['available'] and (download_option['productName'] == 'LAS Product' or
                                                 download_option['productName'] == 'Compressed'):
                data['products'] = download_option['downloadCode']
        download_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/download'
        r = requests.post(download_url, data={'jsonRequest': json.dumps(data)})
        if r.json()['errorCode'] is None:
            data_url = r.json()['data'][0]['url']
            rcode = subprocess.call((['wget', '--background', '--continue', '--progress=bar:force',
                                      '--trust-server-names', '--content-disposition', data_url]))
        else:
            print("Error in downloading: ", r.json()['error'])
    else:
        print("Error in determining download options: ", r.json())
    return rcode


get_api_key()
# noinspection PyArgumentList
data_directories = [x for x in os.listdir() if x.isdigit()]
data_directories.sort(key=lambda x: int(x))  # list in numerical order, instead of '0', '1', '10', '11', etc.
for directory in data_directories:
    # artificially limiting to 110 entries just for initially playing with the data
    if int(directory) > 110:
        continue

    print("Going into directory", directory)
    os.chdir(directory)
    pair_data = json.load(open(directory+'.json', 'r'))
    lidar_rcode = download_data("LIDAR", pair_data[0]['entityId'])
    image_rcode = download_data("NAIP_COMPRESSED", pair_data[1]['entityId'])

    if image_rcode != 0:
        print("Error downloading image for", directory)
        with open('error.log', 'w') as f:
            f.write("Error downloading image")

    if lidar_rcode != 0:
        print("Error downloading lidar for", directory)
        with open('error.log', 'a') as f:
            f.write("Error downloading lidar")

    # download lidar image
    url = pair_data[0]['browseUrl']
    lidar_image_rcode = subprocess.call(['wget', '--background', '--continue', '--progress=bar:force',
                                         '--trust-server-names', '--content-disposition', url])
    if lidar_image_rcode != 0:
        print("Error downloading lidar image for", directory)
        with open('error.log', 'a') as f:
            f.write("Error downloading lidar image")

    # Block to keep number of wget connections low
    while find_number_of_wget_processes() > 7:
        print("Waiting for some wget processes to finish before moving on")
        sleep(3)

    del pair_data  # clean up
    os.chdir('..')
    print("Done with dorectory", directory)
