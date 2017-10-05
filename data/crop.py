
import json
import os

import glymur


def crop(folder_name):
    # Step 1 - Cropping
    # Method:
    # Overview: Get the geo coordinates of the data from the JSON, use those to figure out where, in pixels the lidar
    # data can be found within the image data, and then select just that portion of the image and write it using the
    # glymur lib
    #
    # Details:
    # a) Load the data, and get the 'sceneBounds' which will give the upper right and lower left
    #    coordinates of both the lidar data and the image data in latitude longitude
    # b) Use this data to come up with "normalized" points that reflect where the image should be cropped in a unitless
    #    representation
    # c) Load the image using the glymur library and get its width and height. Multiply those by the normalized points
    #    found in part b, and now we have the pixels which should be cropped out
    # d) Crop and profit

    # a)

    json_filename = folder_name + '/' + folder_name + '.json'
    data = json.load(open(json_filename))
    lidar_points = [float(x) for x in data[0]['sceneBounds'].split(',')]
    image_points = [float(x) for x in data[1]['sceneBounds'].split(',')]

    # b) (Note, implicit assumption being made here that lidar data is entirely within image data)
    dx1 = image_points[2] - image_points[0]
    dx2 = lidar_points[0] - image_points[0]
    dx3 = lidar_points[2] - image_points[0]
    newx1 = dx2/dx1
    newx2 = dx3/dx1

    # Need to measure this "upside-down"
    dy1 = image_points[1] - image_points[3]
    dy2 = lidar_points[3] - image_points[3]
    dy3 = lidar_points[1] - image_points[3]
    newy1 = dy2/dy1
    newy2 = dy3/dy1

    # c)
    # Get the jp2 filename in a fancy way:
    # List all the files in the directory that end in "jp2" and start with "n_", and then take the first element of that
    # list This way we should be able to reliably get the filename for each data directory. I suppose it could also be
    # taken from the JSON metadata. Either approach should be fine
    jp2_filename = [x for x in os.listdir(folder_name) if x[-3:] == 'jp2' and x[:2] == "n_"][0]

    # Open the jp2 file
    jp2_file = glymur.Jp2k(folder_name + '/' + jp2_filename)

    # Get the dimensions of the file
    # noinspection PyUnusedLocal
    [height, width, channels] = jp2_file.shape  # channels is unused
    newwidth1 = int(newx1 * width)
    newwidth2 = int(newx2 * width)
    newheight1 = int(newy1 * height)
    newheight2 = int(newy2 * height)
    print(newwidth1, newwidth2, newheight1, newheight2)
    # In the first run, this came out to an image of 993X1009 pixels. It may be prudent to shrink this down to 950x950
    # since the net needs a consistent input shape, but we'll deal with that after trying a few more

    # d)
    cropped = jp2_file[newheight1:newheight2, newwidth1:newwidth2]
    glymur.Jp2k(folder_name + '/cropped.jp2', data=cropped)


print(os.getcwd())
# noinspection PyArgumentList
data_directories = [x for x in os.listdir() if x.isdigit()]
for directory in data_directories:
    crop(directory)
