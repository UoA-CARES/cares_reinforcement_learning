--8<-- "include/glossary.md"

# Installation Instructions
We recommend using the Stable release versions if you are just looking to use the package directly. If you are seeking to develop the package further then follow the Development Environment instructions for installation.

**Python:** 3.10–3.12 recommended

## Stable Release v3.0.0 (Recommended)
Clone the latest stable release of CARES Reinforcement Learning.

```bash
git clone --branch v3.0.0 https://github.com/UoA-CARES/cares_reinforcement_learning.git

cd cares_reinforcement_learning
pip install -e .[gym]
```

!!! tip "Editable Install"
    This installs the package in editable mode (-e), allowing local changes to the code to be reflected without reinstalling.

!!! tip "Optional dependencies"
    The `[gym]` extra installs common environment dependencies.

    Additional environment packages (e.g. robotics, vision, or MARL environments) may require installing extra dependencies separately.

!!! warning "Unstable Main Branch"
    Clone the **main** branch for the latest features - note this branch may not be stable as it is the working branch.

## Development Environment (UV/pyenv)
We recommend using **pyenv** to manage Python versions and **uv** to manage dependencies and work with reproducible environments from papers. This is because we have various other gym packages that can be installed and used and the general pyenv environment is useful to manage them together. This setup should be used those looking to contribute to the code base or various gym packages.

Clone the latest main of CARES Reinforcement Learning.
```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
```

### 1. Install uv and pyenv

Install `uv` using the official installer:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

!!! tip "Reproducible environments"
    Using `uv` ensures that dependency versions match those used in experiments
    and published results.

    This is recommended when reproducing research or running benchmark comparisons.

Install 'pyenv' using the official installer:

```bash
curl -fsSL https://pyenv.run | bash
```

!!! warning "pyenv setup"
    After installing `pyenv`, ensure it is correctly added to your shell.

    You may need to restart your terminal or run:

    ```bash
    exec $SHELL
    ```

    before `pyenv` commands are available.

### 2. Setup Virtual Environment (pyenv)
Install the required Python version - note you can use 3.12 if you prefer.

```bash
pyenv install 3.10
pyenv virtualenv 3.10 cares_rl_310
pyenv activate cares_rl_310
```

### 3. Install Requirements (UV)
Install the project and its requirements - note we are using the **--active** command to work inside of the pyenv environment.

```bash
cd cares_reinforcement_learning
uv sync --active --extra gym
```

!!! warning "uv environment usage"
    The `--active` flag installs dependencies into the currently active environment.

    Ensure your `pyenv` environment is activated before running `uv sync`.

!!! warning "Installation methods"
    Use either the `pip install -e .[gym]` method **or** the `uv sync` method.

    Do not mix both approaches in the same environment, as this can lead to
    dependency conflicts.

## Verify Installation

After installation, verify that the CLI is available:

```bash
cares-rl -h
```

You should see the help description below if all things are installed correctly. 

```bash
usage: <command> [<args>]

positional arguments:
  {train,evaluate,test,resume}
                        Commands to run this package

options:
  -h, --help            show this help message and exit

```

--8<-- "include/links.md"