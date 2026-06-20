# Dependency Management

## What are the requirements for QUEENS?

Currently, QUEENS is only tested on UNIX systems (Ubuntu and macOS on arm64). Besides Python 3.12
or newer, QUEENS requires [rsync](https://rsync.samba.org/) in order to copy simulation files.

QUEENS declares its Python dependencies in `pyproject.toml`.
For development and CI-like reproducibility, dependencies are managed with Pixi. The Pixi
environments are declared in `pyproject.toml` and locked in `pixi.lock`:

- `default`: core QUEENS dependencies
- `all`: runtime extras without development tools
- `dev`: full contributor setup, including development tools, tutorials, and 4C support

For installation information see the [README.md](https://github.com/queens-py/queens/blob/main/README.md).

## Changing the requirements
For instructions on adding, removing, or updating dependencies, see the
[contributing guide](../contributing.md).
