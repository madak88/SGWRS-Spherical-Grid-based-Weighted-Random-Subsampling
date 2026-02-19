FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set shell 
SHELL ["/bin/bash", "-c"]

# Set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

# Start with root user
USER root

### Create new user
#
# Creating a user inside the container, so we won't work as root.
# Setting all setting all the groups and stuff.
#
###

# Expect build-time argument
ARG HOST_USER_GROUP_ARG
# - create group appuser with id 999
# - create group hostgroup with ID from host. This is needed so appuser can manipulate the host files without sudo.
# - create appuser user with id 999 with home; bash as shell; and in the appuser group
# - change password of appuser to admin so that we can sudo inside the container
# - add appuser to sudo, hostgroup and all default groups
# - copy default bashrc and add ROS sourcing
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/  


### Install the project
#
# If you install multiple project, you should follow the same 
# footprint for each:
# - dependencies
# - pre install steps
# - install
# - post install steps
#
###

# Basic dependencies for everything
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    sudo\
    netbase\
    git\
    build-essential\    
    wget\
    curl\
    gdb\
    unzip\
    ffmpeg\
    libsm6\
    libxext6

RUN pip install\
    h5py\
    plyfile\
    Ninja

# pointpillars dependencies
RUN pip install\
    numba\
    open3d\
    opencv_python\
    PyYAML\
    setuptools\
    tqdm\
    tensorboard\
    pyvista

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

RUN pip install "numpy<2"

# Switching back to appuser, so tha container starts there
USER appuser
RUN export PATH="/home/appuser/.local/bin:$PATH"
WORKDIR /home/appuser
RUN echo 'export PYTHONPATH=/home/appuser/sgwrs:$PYTHONPATH' >> ~/.bashrc
