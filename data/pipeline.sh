#!/bin/bash -xe

trap 'kill $(jobs -p)' EXIT

mkdir -p preprocessing
mkdir -p completed
mkdir -p failed
mkdir -p downloading

# Start the downloader and grab it's PID. We will be watching for it to be done
python3 downloader.py > downloader.log 2>&1 &
DOWNLOADER_PID=$!
echo "Downloader PID: $DOWNLOADER_PID"


# Every 10 minutes or so, check the directories in 'downloading'
# If all the wget-log files in a directory contain the word 'saved', move the directory to the 'preprocessing' folder
# and launch the preprocessing scripts. Once they're done, move the folder to the 'completed' folder
while [ $(ps -p $DOWNLOADER_PID | wc -l) -eq 2 ]; do
    echo "Checking for data and performing relevant actions"
    FOLDERS=$(ls downloading)
    for folder in ${FOLDERS}; do
        # First check for 'saved' in all 3 wget files
        ZIP=0; JP2=0; JPG=0
        if [[ -e downloading/$folder/jpg.wget ]]; then  # if the jpg.wget exists, the other probably do too
            ZIP=$(grep saved downloading/${folder}/*wget | grep ZIP | wc -l)
            JP2=$(grep saved downloading/${folder}/*wget | grep jp2 | wc -l)
            JPG=$(grep saved downloading/${folder}/*wget | grep jpg | wc -l)
        fi
        if [[ ${ZIP} -eq 1 ]] && [[ ${JP2} -eq 1 ]] && [[ ${JPG} -eq 1 ]] && [[ ! -e downloading/$folder/moved_to_preprocessing ]]; then
            echo "Moving $folder data to preprocessing"
            mkdir -p preprocessing/$folder
            mv downloading/${folder}/*.[jZ]* preprocessing/$folder/
            touch downloading/$folder/moved_to_preprocessing  # let downloader know this data has been downloaded
        fi
    done
    # Now that all files ready for pre-processing are staged, launch the preprocessing pipeline
    # List out 20 directories in the pre-processing folder and put them into a file which will be read by the python
    # files. The point of this is to make sure we only process 20 files at a time, to limit RAM and disk usage
    ls preprocessing | head -20 > folders_to_process.txt
    echo "Starting cropping"
    python3 crop.py &
    CROP_PID=$!
    echo "Unzipping and removing. CROP_PID: $CROP_PID"
    python3 unzip_and_remove.py
    echo "Waiting on cropping"
    wait ${CROP_PID}
    echo "Converting LAS to matrix"
    python3 convert_las_to_matrix.py
    echo "Preprocessing done"
    rm folders_to_process.txt
    FOLDERS=$(ls preprocessing)
    # And now that preprocessing is done, move all these files to the completed folder
    for folder in ${FOLDERS}; do
        if [[ -e preprocessing/$folder/failed.txt ]]; then
            echo "Moving $folder to failed"
            mv preprocessing/${folder} failed/
        elif [[ -e preprocessing/$folder/cropped ]] && [[ -e preprocessing/$folder/pickled ]]; then
            echo "Moving $folder to completed"
            mv preprocessing/${folder} completed/
        fi
    done
    echo "Waiting for downloader to load more data..."
    sleep 10
done
