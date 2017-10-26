import json
import os
import requests
import sys
import subprocess
import hashlib
import sys
from time import sleep

DATA_DIR=sys.argv[1]
os.chdir(DATA_DIR)
paired_data = json.load(open('paired_data.json', 'r'))

API_key = "39fa86f1e5454638a9d4bfc78f5e2037"

# Make directories for each pair of data, and put the information for each pair into each folder
os.chdir('downloading')
for i, pair_data in enumerate(paired_data):
    # noinspection PyArgumentList
    hash_input = pair_data[0]['summary'] + '|' + pair_data[1]['summary']
    dirname = hashlib.md5(bytes(hash_input.encode())).hexdigest()[:6]
    os.makedirs(dirname, exist_ok=True)  # TODO: Need to check if this exists in preprocessing or completed
    json_filename = dirname + '/' + dirname + '.json'
    json.dump(pair_data, open(json_filename, 'w'), indent=4)

os.chdir('..')
del paired_data  # clean up


# Now query USGS for download info
# Get the API key
def get_api_key():
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/login'
    print(os.getcwd())
    password = open('password.txt', 'r').readline()
    password = password.split('\n')[0]
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
    # Explanation of command: list all processes | find all containing wget | eliminate the ones containing grep | count
    # lines. The third step gets rid of the process created by the second step from the output of the third step.
    ps = subprocess.Popen("ps -ef | grep wget | grep -v grep | wc -l", shell=True, stdout=subprocess.PIPE)
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
                                      '--trust-server-names', '--content-disposition',
                                      '--output-file',str(entityid) + '.wget', data_url]))
        else:
            print("Error in downloading: ", r.json()['error'])
    else:
        print("Error in determining download options: ", r.json())
    return rcode


get_api_key()
data_directories = os.listdir('downloading')
for directory in data_directories:
    # Ensure that the filesystem has at least a couple dozen GBs of free space left, otherwise, wait.
    # As the preprocessing module runs, space should become free
    while True:
        filesystem_stats = os.statvfs('.')
        free_gbs = filesystem_stats.f_frsize * filesystem_stats.f_bfree / 1024 / 1024 / 1024
        if free_gbs > 30:
            break
        else:
            print("Waiting for space to free up, only seeing", free_gbs, "GB free")
            sleep(3)

    print("Going into directory", directory)
    os.chdir(DATA_DIR + "/downloading/" + directory)
    if os.path.exists('moved_to_preprocessing'):
        continue  # This is a signal that this data has already been downloaded
    pair_data = json.load(open(directory+'.json', 'r'))

    image_rcode = download_data("NAIP_COMPRESSED", pair_data[1]['entityId'])
    if image_rcode != 0:
        print("Error downloading image for", directory)
        with open('error.log', 'w') as f:
            f.write("Error downloading image")

    lidar_rcode = download_data("LIDAR", pair_data[0]['entityId'])
    if lidar_rcode != 0:
        print("Error downloading lidar for", directory)
        with open('error.log', 'a') as f:
            f.write("Error downloading lidar")

    # download lidar image
    url = pair_data[0]['browseUrl']
    lidar_image_rcode = subprocess.call(['wget', '--background', '--continue', '--progress=bar:force',
                                         '--trust-server-names', '--content-disposition',
                                         '--output-file', 'jpg.wget', url])
    if lidar_image_rcode != 0:
        print("Error downloading lidar image for", directory)
        with open('error.log', 'a') as f:
            f.write("Error downloading lidar image")

    # Block to keep number of wget connections low
    while find_number_of_wget_processes() > 7:
        print("Waiting for some wget processes to finish before moving on")
        sleep(3)

    del pair_data  # clean up
    print("Done with directory", directory)

# Once this loop is over, block until all wgets are done
while find_number_of_wget_processes() > 0:
    print("Waiting for some wget processes to finish before exiting")
    sleep(3)

# Note super happy with this solution, as it has some fragility in that it could hang if there are other, unrelated wget
# processes running on the system. Unfortunately, I don't currently see a way of getting a valid handle to the wget
# processes that are launched from this app. PIDs can be obtained, but they can be re-used, so not a valid handle.

