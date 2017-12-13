if [[ $# -ne 1 ]]; then
    echo "$0 <container_name>"
else
    nvidia-docker run -it --rm --privileged -v ~/tnn_data:/home/docker/data -v ~/topography_neural_net/:/home/docker/code --name $1 nbelakovski/tnn:gpu
fi
