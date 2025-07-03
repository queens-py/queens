# 4C

This package contains drivers and utilities realted to the for multiphysics code [4C](https://github.com/4C-multiphysics/4C). We like them and we work closely with them.

## Material random field interface
In order to create random material field in combintation with 4C, we require the package [fourcipp](https://github.com/4C-multiphysics/fourcipp). Therefore install QUEENS via (in the QUEENS main directory):
```
pip install -e .[dev,fourc]
```
(You can omit the `dev` if you don't require the additional development packages).
