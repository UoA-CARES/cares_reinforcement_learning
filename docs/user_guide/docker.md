# Docker

CARES Reinforcement Learning provides a Docker option for reproducible training environments, GPU-enabled experiments, and CARES HPC deployment.

The Docker image installs the CARES RL package inside a Linux container with the system dependencies required for common RL environments such as Gymnasium, MuJoCo, OpenGL/EGL rendering, and video output.

## When to use Docker

Use Docker when you want:

- a reproducible training environment
- a clean install that does not modify your host Python environment
- GPU support for training and rendering
- a portable image for CARES HPC workers or remote machines

For normal development, a local editable install is still recommended.

---

## Build the image

From the root of the repository:

```bash
docker build -t cares-rl:main .
```

This builds the image using the default CARES RL branch configured in the Dockerfile.

---

### Build a specific version

The Dockerfile supports selecting a branch, tag, or commit using the `CARES_RL_REF` build argument.

Use `main` for the latest main branch:

```bash
docker build --build-arg CARES_RL_REF=main -t cares-rl:main .
```

Use release tags for reproducible user-facing images:

```bash
docker build --build-arg CARES_RL_REF=v0.1.0 -t cares-rl:v0.1.0 .
```

Use commit SHAs for exact experiment reproduction:

```bash
docker build --build-arg CARES_RL_REF=abc1234 -t cares-rl:abc1234 .
```

---


## Run Docker Container
After building, you can run the container and check that the `cares-rl` CLI is available:

```bash
docker run --rm -it cares-rl:main cares-rl --help
```

Below is an example of running a training command inside the container with GPU support and mounted output directories:

```bash
docker run --rm -it \
  --user $(id -u):$(id -g) \
  --gpus all \
  -v "$PWD/outputs:/workspace/output" \
  -v "$PWD/datasets:/workspace/datasets:ro" \
  cares-rl:main cares-rl train cli --gym dmcs --domain ball_in_cup --task catch SAC
```

!!! note "Full Commands"
    The above is an example of a full command with GPU support and mounted volumes.

    You can modify the `cares-rl train cli` command and its arguments as needed for your experiments.

    Ensure that the `--gpus all` flag is included to enable GPU support, and adjust the volume mounts (`-v`) to point to your desired output and dataset directories on the host machine.

!!! note "cares-rl Usage"
    The `cares-rl` CLI commands and arguments are the same inside the container as they are in a local installation.

    You can run any `cares-rl` command inside the container, such as `train`, `evaluate`, `test`, or `resume`, with the same syntax as you would on your host machine.

    Please see the [Experiment Guide](experiment.md) section for more details on using the `cares-rl` CLI.


The instructions below explain the full command for running the container with GPU support and mounting output directories.

### Run with GPU support

To use NVIDIA GPUs, install the NVIDIA Container Toolkit on the host machine.

Then run:

```bash
docker run --rm -it \
  --gpus all \
  cares-rl:main
```

Check that PyTorch can see the GPU:

```bash
docker run --rm -it \
  --gpus all \
  cares-rl:main \
  python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

---

### Mount outputs to the host

The container writes logs and experiment outputs to:

```text
/workspace/output
```

The Dockerfile sets:

```bash
CARES_LOG_BASE_DIR=/workspace/output
```

To save results on the host machine, mount a local folder:

```bash
mkdir -p ~/cares_rl_outputs

docker run --rm -it \
  --gpus all \
  -v "$HOME/cares_rl_outputs:/workspace/output" \
  cares-rl:main
```

After the run, outputs will appear on the host system in:

```text
$HOME/cares_rl_outputs
```

!!! note "Mounting Training Data"
    If you require training data or datasets, you can also mount additional directories into the container. For example, to mount a local `cares_rl_logs` folder for `eval` or `test` commands that require access to training logs:

---

### File ownership and permissions

By default, Docker runs processes as the user configured inside the container. When mounting host directories, this can result in output files being owned by a different user than the one that launched the container.

To ensure output files are owned by the current host user, pass your user and group IDs when running the container:

```bash
docker run --rm -it \
  --user $(id -u):$(id -g) \
  --gpus all \
  -v "$HOME/cares_rl_outputs:/workspace/output" \
  cares-rl:main
```

Files written to mounted directories will be owned by your host account.
This is particularly useful when:

- running experiments locally
- sharing output directories between containers and the host
- using network storage or NAS mounts
- developing on multi-user Linux systems

## Example full workflow

```bash
docker build \
  --build-arg CARES_RL_REF=main \
  -t cares-rl:main .

mkdir -p outputs datasets

docker run --rm -it \
  --user $(id -u):$(id -g) \
  --gpus all \
  -v "$PWD/outputs:/workspace/output" \
  -v "$PWD/datasets:/workspace/datasets:ro" \
  cares-rl:main
```

Outputs written inside the container to:

```text
/workspace/output
```

will be available on the host at:

```text
./outputs
```