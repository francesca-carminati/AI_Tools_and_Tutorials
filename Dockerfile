FROM continuumio/miniconda3:latest AS miniconda
FROM nvidia/cuda:11.1-cudnn8-runtime

ARG USER_ID
ARG USER_NAME
ARG NEPTUNE_API_TOKEN
ARG NEPTUNE_PROJECT

ENV NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN}
ENV NEPTUNE_PROJECT=${NEPTUNE_PROJECT}
ENV WORKDIR=/workspace
WORKDIR $WORKDIR
COPY . $WORKDIR/

COPY --from=miniconda /opt/conda /opt/conda

RUN apt-get update && apt-get install -y git sudo vim wget; apt-get clean

# Fix permissions
RUN useradd --create-home --uid ${USER_ID} ${USER_NAME}; \
    adduser ${USER_NAME} sudo; \
    echo "${USER_NAME} ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers; \
    chown ${USER_NAME} $WORKDIR; \
    chown -R ${USER_NAME}: /opt/conda

# Configure for huggingface and deep learning frameworks
RUN mkdir /.local && chmod 0777 /.local; \
    mkdir /.jupyter && chmod 0777 /.jupyter; \
    mkdir /.cache && chmod 0777 /.cache; \
    mkdir /.config && chmod 0777 /.config; \
    touch /.netrc && chmod 0777 /.netrc

USER ${USER_NAME}

# Install python dependencies
ENV PATH="/opt/conda/bin:${PATH}"
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN conda install mamba -n base -c conda-forge
RUN mamba env create --file environment.yml
RUN echo "conda activate Example" >> ~/.bashrc
