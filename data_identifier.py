#!/usr/local/bin/python3

import json
import requests
import sys
from math import ceil, floor

# Some globals

API_key = "436411e433824075ae241eb8abc83824"
# west_coast_sw_lat = 35.0
# west_coast_sw_lon = -125.0
# west_coast_ne_lat = 49.0
# west_coast_ne_lon = -119.0
west_coast_sw_lat = 48
west_coast_sw_lon = -121.4
west_coast_ne_lat = 48.4
west_coast_ne_lon = -120.7


def create_lat_lon_dict(lat, lon):
    return {"latitude": lat, "longitude": lon}


def create_spatial_filter(ll_lat, ll_lon, ur_lat, ur_lon):
    corners = {"lowerLeft": create_lat_lon_dict(ll_lat, ll_lon), "upperRight": create_lat_lon_dict(ur_lat, ur_lon)}
    spatial_filter = {"filterType": "mbr"}
    return {**spatial_filter, **corners}  # merge the two dicts


# Get the API key
def get_api_key():
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/login'
    password = open('password.txt', 'r').readline()
    password = password.split('\n')[0]
    data = {"username": "nbelakovski", "password": password, "authType": "EROS", "catalogId": "EE"}
    r = requests.post(login_url, data={'jsonRequest': json.dumps(data)})
    if r.json()['errorCode'] is None:
        global API_key
        API_key = r.json()['data']
    else:
        print("Failure to obtain API key, errorCode: ", r.json()['errorCode'])
        sys.exit(-1)

    print(API_key)


# (Exploratory) Get all dataset names
def get_dataset_names():
    datasets_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/datasets'
    data = {"apiKey": API_key}
    r = requests.post(datasets_url, data={'jsonRequest': json.dumps(data)})

    if r.json()['errorCode'] is None:
        for item in r.json()['data']:
            print(item['datasetName'], ", ", item['datasetFullName'])
    else:
        print("Failure to obtain datasets, errorCode: ", r.json()['errorCode'])
        sys.exit(-2)


# Get topography data for specified region
def get_topo_data():
    datasets_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/search'
    data = {"apiKey": API_key,
            "datasetName": "LIDAR",
            "spatialFilter":
                create_spatial_filter(west_coast_sw_lat, west_coast_sw_lon, west_coast_ne_lat, west_coast_ne_lon),
            "maxResults": 1000}
    r = requests.post(datasets_url, data={'jsonRequest': json.dumps(data)})

    if r.json()['errorCode'] is None:
        results = r.json()['data']['results']
        print("Found", len(results), "pieces of data for specified coordinates")
        return results
    else:
        print("Failure to obtain datasets, errorCode: ", r.json()['errorCode'])
        sys.exit(-3)


# This function should calculate the overlap of two rectangles given 4 coordinates
# Input is two arrays, each containing the lower left and upper right coordinate points
# Despite the fact that these are latitudes and longitudes in non-euclidean space, we
# will make the flat world assumption given that the region is small enough to be well
# approximated by euclidean geometry
def calculate_overlap(rect1, rect2):
    # min of maxes - max of mins
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    if dx >= 0 and dy >= 0:
        return dx*dy
    else:
        return 0



# Get image data for each element of the topographical data
# This function should return input/output pairs with enough information for downloading
def get_image_data(input_topo_data):
    datasets_url = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/search'
    paired_data = []
    for lidar_scene in input_topo_data:
        # First get the corners of the lidar data, and truncate them so as to make sure only limited DOQ data is
        # returned
        corners = lidar_scene['sceneBounds'].split(',')
        ll_lon = floor(float(corners[0]) * 100) / 100
        ll_lat = ceil(float(corners[1]) * 100) / 100
        ur_lon = ceil(float(corners[2]) * 100) / 100
        ur_lat = floor(float(corners[3]) * 100) / 100
        data = {"apiKey": API_key, "datasetName": "NAIP_COMPRESSED",
                "spatialFilter": create_spatial_filter(ll_lat, ll_lon, ur_lat, ur_lon), "maxResults": 15}
        r = requests.post(datasets_url, data={'jsonRequest': json.dumps(data)})

        if r.json()['errorCode'] is None:
            # If multiple results are returned, go through and calculate the one with the largest overlap. We will
            # crop the topography and image data to fit to the overlap, provided the resultant area is above some
            # minimum Other concern: if several images have a similar overlap, i.e. within 10%, and all are above the
            # minimum, we should take the one that was recorded closest to the LIDAR data in time
            images = r.json()['data']['results']
            max_overlap = 0
            input_image = None
            for image_metadata in images:
                # calculate overlap
                rect1 = [float(x) for x in image_metadata['sceneBounds'].split(',')]
                rect2 = [float(x) for x in lidar_scene['sceneBounds'].split(',')]
                overlap = calculate_overlap(rect1, rect2)
                if overlap > max_overlap:
                    input_image = image_metadata
                    max_overlap = overlap
            # Once the proper image is determined, add its download information to the lidar_scene
            data_pair = (lidar_scene, input_image)
            paired_data.append(data_pair)
        else:
            print("Failure to obtain datasets, return string: ", r.json())

        if len(paired_data) % 10 == 0:
            print("Found", len(paired_data), "corresponding images")

    return paired_data


get_api_key()
print("=======================")
print("Getting topography data")
print("=======================")
topo_data = get_topo_data()
# download_data(topo_data)
# for image data, should floor/ceil the lidar data corners at the hundredths place
print("=================")
print("Getting image data")
print("=================")
paired_data_to_download = get_image_data(topo_data)

with open('data/paired_data.json', 'w') as f:
    json.dump(paired_data_to_download, f, indent=4)
