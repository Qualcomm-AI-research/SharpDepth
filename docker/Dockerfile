FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        git \
        less \
	libgl1 \
	libglib2.0-0 \
	passwd \
        python-is-python3 \
        python3 \
        python3-pip \
        tree \
        vim \
	wget \
	&& \
    rm -rf /var/lib/apt/lists/*


RUN groupadd -g ${GID} ${USERNAME} && useradd -m -u ${UID} -g ${GID} ${USERNAME}


ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh  && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && $CONDA_DIR/bin/conda clean -a -y

USER ${USERNAME}
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY docker/requirements.txt requirements.txt
RUN conda create -n sharpdepth python=3.10 && conda clean -a
RUN conda run -n sharpdepth pip install -v -r requirements.txt

# Install UniDepth

RUN git clone https://github.com/lpiccinelli-eth/UniDepth.git /tmp/UniDepth
RUN cd /tmp/UniDepth && conda run -n sharpdepth pip install -v -e . --no-deps
RUN cd /tmp/UniDepth/unidepth/ops/knn && conda run -n sharpdepth pip install -v -e . --no-deps

ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate sharpdepth && exec \"$@\"", "--"]
CMD ["bash"]
