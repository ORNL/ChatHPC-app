# Python Project Template

Documentation: <https://devdocs.ornl.gov/7ry/python-project-template>  
Coverage Report: <https://devdocs.ornl.gov/7ry/python-project-template/coverage>

# Quick Start with Template

1. Fork the repository.
2. Run `setup_template.sh` to setup the template for the new project.
3. Remove `setup_template.sh`

Note: If you are using gitlab and the runners are setup for the group/project and the entries into setup_template.sh are correct, the CI pipeline will start building and deploying the documentation to devdocs. There is currently a known bug with the first run of the pipeline where deploy coverage depends on deploy docs running first. If you enconder an rsync error with deploy coverage try rerunning it after deploy docs finishes.

----

**Table of Contents**
- [Python Project Template](#python-project-template)
- [Quick Start with Template](#quick-start-with-template)
    - [Installation](#installation)
    - [Setup pre-commit Git hooks](#setup-pre-commit-git-hooks)
    - [Running with hatch](#running-with-hatch)
    - [Testing with hatch](#testing-with-hatch)
    - [Format code with hatch](#format-code-with-hatch)
    - [View version with hatch](#view-version-with-hatch)
    - [Update version with hatch](#update-version-with-hatch)
    - [Documentation](#documentation)
        - [Commands](#commands)
        - [Hatch Commands](#hatch-commands)

## Installation

For development in folder:

```bash
git clone git@code.ornl.gov:7ry/python-project-template.git
cd python-project-template
python3 -m venv --upgrade-deps --prompt $(basename $PWD) .venv
source .venv/bin/activate
pip install -e .
```

For use in virtual environment:

```bash
python3 -m venv --upgrade-deps --prompt $(basename $PWD) .venv
source .venv/bin/activate
pip install git+ssh://git@code.ornl.gov/7ry/python-project-template.git
```

## Setup pre-commit Git hooks

Use hatch or install pre-commit inside python virtual environment.
```bash
hatch shell
```
or
```bash
pip install pre-commit
```

Then install the hooks.
```bash
pre-commit install
```

Note: You might have to upgrade pre-commit.
```bash
pre-commit autoupdate
```

Note: The markdown linter requires Ruby gem to be installed to auto-install and run mdl.

On Ubuntu this can be done with:
```bash
sudo apt install ruby-full
```

## Running with hatch

```bash
hatch shell
```

## Testing with hatch

```bash
hatch run test
```

To test on all python versions:
```bash
hatch run all:test
```

Run tests and print the output.
```bash
hatch run test -v -s
```

## Format code with hatch

```bash
hatch fmt
```

Update default ruff rules:

```bash
hatch fmt --check --sync
```

## View version with hatch

```bash
hatch version
```

## Update version with hatch

```bash
hatch version <new version>
```

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org) using the [Read the Docs](https://docs.readthedocs.io/en/stable/) theme.

### Commands

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.

Other useful commands:
- `mkdocs serve -a 0.0.0.0:8000` - Serve with extenal access to the site. (Useful in ExCL to view using foxyproxy.)

### Hatch Commands

View environment
```bash
hatch env show docs
```

Build documentation
```bash
hatch run docs:build
```

Serve documentation
```bash
hatch run docs:serve
```
or
```bash
hatch run docs:serve -a 0.0.0.0:8000
```
