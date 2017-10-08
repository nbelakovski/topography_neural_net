import json
import os
import requests
from tqdm import tqdm

paired_data = json.load(open('paired_data.json', 'r'))

API_key = "39fa86f1e5454638a9d4bfc78f5e2037"

# Make directories for each pair of data, and put the information for each pair into each folder
for i, data in enumerate(paired_data):
    # noinspection PyArgumentList
    os.makedirs(name=str(i), exist_ok=True)
    json_filename = str(i) + '/' + str(i) + '.json'
    json.dump(data, open(json_filename, 'w'), indent=4)

# Now query USGS for download info
# Get the API key
def get_api_key():
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/login'
    password = open('../password.txt','r').readline()
    data = {"username": "nbelakovski", "password": password, "authType": "EROS", "catalogId": "EE"}
    r = requests.post(login_url, data={'jsonRequest': json.dumps(data)})
    if r.json()['errorCode'] is None:
        global API_key
        API_key = r.json()['data']
    else:
        print("Failure to obtain API key, errorCode: ", r.json()['errorCode'])
        sys.exit(-1)

    print(API_key)

def download_data(dataset, entityids):
    downloadoptions_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/downloadOptions'
    data = {"apiKey": API_key, "datasetName": dataset, "entityIds": entityids}
    r = requests.post(downloadoptions_url, data={'jsonRequest': json.dumps(data)})
    if r.json()['errorCode'] is None:
        print(r.json()['data'])
        print(r.json()['error'])
        data['products'] = []
        for downloadOptions in r.json()['data']:
            for scene in downloadOptions['downloadOptions']:
                if scene['available'] and scene['downloadCode'] == 'STANDARD':
                    data['products'].append(scene['downloadCode'])
        download_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/download'
        r = requests.post(download_url, data={'jsonRequest': json.dumps(data)})
        if r.json()['errorCode'] is None:
            print(r.json()['error'])
            for i, downloadurl in enumerate(r.json()['data']):
                file = requests.get(downloadurl['url'], stream=True)
                total_size = int(file.headers.get('content-length', 0))
                with open('output.bin', 'wb') as f:
                    for data in tqdm(file.iter_content(), total=total_size, unit='B', unit_scale=True):
                        f.write(data)
        else:
            print("Error in downloading: ", r.json()['error'])
    else:
        print("Error in determining download options: ", r.json())



get_api_key()
topo_entityids = []
image_entityids = []
for scene in paired_data:
    topo_entityids.append(scene[0]['entityId'])
    image_entityids.append((scene[1]['entityId']))

# Probably want to CD into different folders for this operation
download_data("LIDAR", topo_entityids)
download_data("NAIP_COMPRESSED", image_entityids)