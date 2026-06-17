FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# CARES HPC output path for sending result back to the host machine
ENV CARES_LOG_BASE_DIR=/workspace/output

# Headless stuff for MuJoCo and OpenGL
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

ARG USER_NAME=rlt
ARG USER_ID=1000
ARG GROUP_ID=1000

ARG CARES_RL_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libglfw3 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    libopenmpi-dev \
    libsdl2-dev \
    swig \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${GROUP_ID} ${USER_NAME} && \
    useradd \
    --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    --create-home \
    --shell /bin/bash \
    ${USER_NAME}

RUN mkdir -p \
    /home/${USER_NAME}/workspace \
    /workspace/output \
    /workspace/datasets

RUN chown -R ${USER_NAME}:${USER_NAME} \
    /home/${USER_NAME} \
    /workspace

USER ${USER_NAME}

WORKDIR /home/${USER_NAME}/workspace

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

RUN git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git \
    /home/${USER_NAME}/workspace/cares_reinforcement_learning && \
    cd /home/${USER_NAME}/workspace/cares_reinforcement_learning && \
    git checkout ${CARES_RL_REF} && \
    python3 -m pip install --user --upgrade pip && \
    python3 -m pip install --user -e ".[gym]"

WORKDIR /home/${USER_NAME}/workspace/cares_reinforcement_learning

CMD ["bash"]