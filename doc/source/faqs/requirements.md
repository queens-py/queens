# Dependency Management

## What are the requirements for QUEENS?

Currently, QUEENS is only tested on UNIX systems. Besides Python, QUEENS requires [rsync](https://rsync.samba.org/) in order to copy simulation files.

The Python dependencies for QUEENS are managed through the conda environment files in the repository
root:

- `environment.base.yml`
- `environment.dev.yml`
- `environment.tutorials.yml`
- `environment.fourc.yml`

For reproducible installs, QUEENS also provides the composed lock file `composed.conda-lock.yml`.

For more information see the [README.md](https://github.com/queens-py/queens/blob/main/README.md).
