Teaching computer to recognize railroads on maps



# Instructions for installing deep learning environment on Microsoft Azure

### 1)
Install nvidia drivers (if necessary). Might have to check the most recent version
Check for latest drivers 
!!! MAKE SURE YOUR KERNEL VERSION IS GREATER THAN 4.4.0-75 (IT HAS TO BE AT LEAST 4.4.0-77) !!!

(~12 minutes on NC6 [2017-04-25])
```
DRIVER_VERSION="375.66"
wget http://us.download.nvidia.com/tesla/${DRIVER_VERSION}/nvidia-diag-driver-local-repo-ubuntu1604_${DRIVER_VERSION}-1_amd64.deb

sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_${DRIVER_VERSION}-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers

rm nvidia-diag-driver-local-repo-ubuntu1604_${DRIVER_VERSION}-1_amd64.deb

# Optional - restart your machine
# Optional - check the status of GPUs: nvidia-smi
```


### 2) Install Docker
https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository
(~1 minute on NC6 [2017-04-25])

```
# Install packages to allow `apt` to use a repository over HTTPS:
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


# Verify that the key fingerprint is `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`
sudo apt-key fingerprint 0EBFCD88


# Set up apt repo:

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   
sudo apt-get update

# Install latest version of Docker:
sudo apt-get install -y docker-ce
```

### 3) Install nvidia-docker and nvidia-docker-plugin
This automatically sets up all paths/libraries/environment variables necessary for a container started by `sudo nvidia-docker run ...` to access host's gpu(s)

```
LATEST_VERSION="1.0.1"
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v${LATEST_VERSION}/nvidia-docker_${LATEST_VERSION}-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

### 4) Pull docker image from docker-hub

```
sudo nvidia-docker pull spatialcomputing/deep-learning-env-gpu
```

### 5) Run docker environment
```
sudo nvidia-docker run --rm -ti -p 8888:8888 -v /datadrive:/datadrive spatialcomputing/deep-learning-env-gpu
```
If you are not planning on running jupyter notebook from within the container, you don't need to pass `-p host_port:container_port`.
If you are not planning on accessing host's file system, you don't need to pass `-v host_directory_name:how_that_directory_will_appear_within_container_name`, but remember that all files/data generated within running container will be lost when you stop it, unless it was written to host's directory. 
