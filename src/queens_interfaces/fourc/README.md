# 4C

This package contains drivers and utilities related to the multiphysics code
[4C](https://github.com/4C-multiphysics/4C). We like them and we work closely with them.

## Material random field interface
In order to create random material fields in combination with 4C, QUEENS requires the package
[fourcipp](https://github.com/4C-multiphysics/fourcipp). For a regular source install, use the
`fourc` extra:

```bash
python -m pip install ".[fourc]"
```

For Pixi project-based workflow, use the `all` environment
that includes the 4C interface dependencies without development tools:
```bash
pixi install --environment all
pixi run -e all install-editable
```
For development, use the `dev` Pixi environment that includes the `fourc` feature.

For more setup details, see the top-level [README.md](https://github.com/queens-py/queens/blob/main/README.md).
