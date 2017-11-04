import multiprocessing
import json
import os
import sys
import glymur


def crop(folder_name):
    os.chdir(folder_name)
    if os.path.exists('cropped.jp2'):
        return
    data_hash = folder_name.split('/')[-1]
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

    json_filename = data_hash + '.json'
    data = json.load(open(json_filename))
    lidar_points = [float(x) for x in data[0]['sceneBounds'].split(',')]
    image_points = [float(x) for x in data[1]['sceneBounds'].split(',')]
    
    # Check that the lidar is entirely within the image
    latitude_check = image_points[1] < lidar_points[1] < lidar_points[3] < image_points[3]
    longitude_check = image_points[0] < lidar_points[0] < lidar_points[2] < image_points[2]
    if not latitude_check or not longitude_check:
        with open('failed.txt','w') as f:
            f.write('lidar out of bounds.')
        with open('lidar_oob','w') as f:  # make a separate empty file to make it easier to see the cause of failure upon manual investigation
            f.write('')
        return
    
    # b)
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
    jp2_filename = [x for x in os.listdir(folder_name) if x[-3:] == 'jp2' and (x[:2] == "n_" or x[:2] == "m_")][0]

    # Open the jp2 file
    jp2_file = glymur.Jp2k(jp2_filename)

    # Get the dimensions of the file
    [height, width, channels] = jp2_file.shape
    newwidth1 = max(0, int(newx1 * width))
    newwidth2 = min(width, int(newx2 * width))
    newheight1 = max(0, int(newy1 * height))
    newheight2 = min(height, int(newy2 * height))
    # In the first run, this came out to an image of 993X1009 pixels. It may be prudent to shrink this down to 950x950
    # since the net needs a consistent input shape, but we'll deal with that after trying a few more

    # d)
    cropped = jp2_file[newheight1:newheight2, newwidth1:newwidth2, :3]  # only get the first 3 channels, no need for alpha
    print(jp2_file.shape)
    print(cropped.shape)
    try:
        newfile = glymur.Jp2k('cropped.jp2', data=cropped)
    except Exception as e:
        with open('failed.txt', 'a') as f:
            f.write(str(e))
        return

    with open('cropped', 'w') as f:  # This file indicates success to the pipeline processor
        f.write('%d,%d,%d' % tuple(newfile.shape))


with open(sys.argv[1], 'r') as f:
    directories = f.read().splitlines()
with multiprocessing.Pool(3) as p:
    p.map(crop, directories)
