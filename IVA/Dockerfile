
FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="Craig Citro <craigcitro@google.com>"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        cmake \
        wget \
        make \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
        redis-server \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
        apt-get update && \
        apt-get install libnvinfer4=4.1.2-1+cuda9.0

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN apt-get update && apt-get install -y \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libav-tools  \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libjasper-dev \ 
    python-numpy \ 
    python-pycurl \ 
    python-opencv

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        numpy \
        pandas \
        flask \
        redis \
        opencv-python \
        opencv-contrib-python \
        && \
    python3 -m ipykernel.kernelspec

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip3 --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version.
RUN pip3 --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp35-cp35m-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

RUN pip3 --no-cache-dir install \
              keras \
              redis 


#RUN redis-server&

ADD ./r_iva /apps

WORKDIR /apps

CMD ["ls"]

CMD ["python3","/apps/server.py"]

EXPOSE 5000


