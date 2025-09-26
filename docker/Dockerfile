# Select your ROS version or container to start from
FROM osrf/ros:humble-desktop-full

# Some basic tools that you will include everytime
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y  git \
    tmux \
    tmuxinator \
    vim \
    xterm \
    curl \
    ssh \
    tree \
    python3-tk \
    nano \
    libtool \
    libtool-bin \
    htop \
    gdb \
    net-tools \
    build-essential \
    cmake \
    lsb-release \
    iputils-ping

# Enable scroll in tmux
RUN printf "set -g mouse on\n\
bind -n M-Left select-pane -L\n\
bind -n M-Right select-pane -R\n\
bind -n M-Up select-pane -U\n\
bind -n M-Down select-pane -D\n\
\n\
# split panes using | and -\n\
bind | split-window -h\n\
bind - split-window -v\n\
unbind '\"'\n\
unbind %%\n" > /root/.tmux.conf

## ROS2 utils
RUN apt-get install -y \
    python3-rosdep \
    python3-setuptools \
    python3-pip \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    ros-dev-tools

# Add the source for the system ROS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

RUN apt-get install -y \
    ros-humble-aerostack2 \
    ros-humble-tf-transformations

RUN rm -rf log
WORKDIR /root/