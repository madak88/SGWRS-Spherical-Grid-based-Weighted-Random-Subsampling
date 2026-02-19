#!/bin/bash

# Parameters
container_name="sgwrs"
image_name="sgwrs"
image_tag="latest"

src_folder=$(pwd)/sgwrs

# Check if container exists
if [[ $( docker ps -a -f name=$container_name | wc -l ) -eq 2 ]];
then
    echo "Container already exists. Do you want to restart it or remove it?"
    select yn in "Restart" "Remove"; do
        case $yn in
            Restart )
                echo "Restarting it... If it was started without USB, it will be restarted without USB.";
                docker restart $container_name;
                break;;
            Remove )
                echo "Stopping it and deleting it... You should simply run this script again to start it.";
                docker stop $container_name;
                docker rm $container_name;
                break;;
        esac
    done
else
    echo "Container does not exist. Creating it."
    # NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES sets the visible devices and capabilities of the GPU
    # gpus all adds all the gpus to the container
    # runtime=nvidia tells the docker engine to use the nvidia runtime
    # these parameters are needed for the gpu to work properly inside the container
    docker run \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --volume $src_folder:/home/appuser/sgwrs \
        --volume /home/madak32/Desktop/SGWRS_Spherical-Grid-based-Weighted-Random-Subsampling/sgwrs_ds/kitti_data:/home/appuser/sgwrs/kitti_data \
        --interactive \
        --tty \
        --detach \
        --gpus all \
        --runtime=nvidia \
        --name $container_name \
        --shm-size=512m \
        $image_name:$image_tag 
fi
