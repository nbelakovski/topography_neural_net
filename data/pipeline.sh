#!/bin/bash -xe

NUMARGS=$#
if [[ $NUMARGS -ne 2 ]]; then
    echo "Usage: ./pipeline.sh DATA_DIRECTORY DATA_FILENAME"
    exit
fi

trap 'kill $(jobs -p)' EXIT

CWD=$(pwd)
DATA_DIR=$1
cd $DATA_DIR
mkdir -p preprocessing
mkdir -p completed
mkdir -p failed
mkdir -p downloading
cd $CWD

DATA_FILENAME=$2
# Start the downloader and grab it's PID. We will be watching for it to be done
python3 downloader.py $DATA_DIR $DATA_FILENAME > $DATA_DIR/downloader.log 2>&1 &
DOWNLOADER_PID=$!
echo "Downloader PID: $DOWNLOADER_PID"

rm -f folders_to_process.txt
rm -f remaining_folders_to_process.txt
# Every 10 minutes or so, check the directories in 'downloading'
# If all the wget-log files in a directory contain the word 'saved', move the directory to the 'preprocessing' folder
# and launch the preprocessing scripts. Once they're done, move the folder to the 'completed' folder
while [[ $(ps -p $DOWNLOADER_PID | wc -l) -eq 2 ]] || [[ $(ls $DATA_DIR/preprocessing | wc -l ) -gt 0 ]]; do
    echo "Checking for data and performing relevant actions"
    FOLDERS=$(ls $DATA_DIR/downloading)
    for folder in ${FOLDERS}; do
        # First check for 'saved' in all 3 wget files
        ZIP=0; JP2=0; JPG=0
        if [[ -e $DATA_DIR/downloading/$folder/jpg.wget ]]; then  # if the jpg.wget exists, the other probably do too
            ZIP=$(grep saved $DATA_DIR/downloading/${folder}/*wget | grep ZIP | wc -l)
            JP2=$(grep saved $DATA_DIR/downloading/${folder}/*wget | grep jp2 | wc -l)
            JPG=$(grep saved $DATA_DIR/downloading/${folder}/*wget | grep jpg | wc -l)
        fi
        if [[ ${ZIP} -eq 1 ]] && [[ ${JP2} -eq 1 ]] && [[ ${JPG} -eq 1 ]] && [[ ! -e $DATA_DIR/downloading/$folder/moved_to_preprocessing ]]; then
            echo "Moving $folder data to preprocessing"
            mkdir -p $DATA_DIR/preprocessing/$folder
            mv $DATA_DIR/downloading/${folder}/*.[jZ]* $DATA_DIR/preprocessing/$folder/
            touch $DATA_DIR/downloading/$folder/moved_to_preprocessing  # let downloader know this data has been downloaded
        fi
    done
    # Now that all files ready for pre-processing are staged, launch the preprocessing pipeline
    # List out 20 directories in the pre-processing folder and put them into a file which will be read by the python
    # files. The point of this is to make sure we only process 20 files at a time, to limit RAM and disk usage
    find $DATA_DIR/preprocessing -maxdepth 1 -mindepth 1 -type d | head -20 > folders_to_process.txt
    if [[ $(wc -l folders_to_process.txt | cut -d' ' -f 1) -lt 2 ]]; then
         echo "Nothing to process, continuing"
         sleep 10
         continue
    fi
    echo "Starting cropping"
    python3 crop.py folders_to_process.txt &
    CROP_PID=$!
    echo "Unzipping and removing. CROP_PID: $CROP_PID"
    python3 unzip_and_remove.py folders_to_process.txt
    echo "Waiting on cropping"
    wait ${CROP_PID}
    # Some folders have shown issues where their ZIP does not appear to contain an LAS. Look for those folders and move them to failed
    # Also note which folder are successful, so that the new list can be passed to convert_last_to_matrix.py.
    FOLDERS=$(cat folders_to_process.txt | awk -F'/' '{print $6}')
    for FOLDER in ${FOLDERS}; do
        LAS=0; ZIP=0
        if ls $DATA_DIR/preprocessing/$FOLDER/*las > /dev/null 2>&1; then
            LAS=1
        fi
        if ls $DATA_DIR/preprocessing/$FOLDER/*ZIP > /dev/null 2>&1; then
            ZIP=1
        fi
        if [[ $LAS -eq 0 ]] && [[ $ZIP -eq 0 ]]; then
             echo "issues with ZIP/LAS" >> $DATA_DIR/preprocessing/$FOLDER/ziplas_issues.txt
             mv $DATA_DIR/preprocessing/$FOLDER $DATA_DIR/failed/
        else
             echo $DATA_DIR/preprocessing/$FOLDER >> remaining_folders_to_process.txt
        fi
    done
    echo "Converting LAS to matrix"
    if [[ -e remaining_folders_to_process.txt ]] ; then
        python3 convert_las_to_matrix.py remaining_folders_to_process.txt
        echo "Preprocessing done"
        rm remaining_folders_to_process.txt
    fi
    rm folders_to_process.txt
    FOLDERS=$(ls $DATA_DIR/preprocessing)
    # And now that preprocessing is done, move all these files to the completed folder
    for folder in ${FOLDERS}; do
        if [[ -e $DATA_DIR/preprocessing/$folder/failed.txt ]]; then
            echo "Moving $folder to failed"
            mv $DATA_DIR/preprocessing/${folder} $DATA_DIR/failed/
        elif [[ -e $DATA_DIR/preprocessing/$folder/cropped ]] && [[ -e $DATA_DIR/preprocessing/$folder/pickled ]]; then
            echo "Moving $folder to completed"
            mv $DATA_DIR/preprocessing/${folder} $DATA_DIR/completed/
        fi
    done
    echo "Waiting for downloader to load more data..."
    sleep 10
done
