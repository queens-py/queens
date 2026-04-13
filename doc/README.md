# :book: HTML documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/#) to generate the [QUEENS documentation](https://queens-py.github.io/queens). It automatically builds the html documentation from the docstrings.

We believe that documentation is essential and therefore welcome any improvements :blush:

## :woman_teacher: Build the documentation

To build the documentation, you first need to set up a QUEENS environment as described in the [README.md](../README.md).
Make sure to install the packages for QUEENS development and tutorial.
Next, register the environment as a Jupyter kernel with:

```bash
python -m ipykernel install --user --name queens --display-name "Python (queens)"
```

When building the documentation on your machine for the first time or after adding new modules or classes to QUEENS, one needs to first rebuild the `autodoc index` by running:

```bash
cd <queens-base-directory>
sphinx-apidoc -o doc/source src/ -fMT
```

To actually build the html-documentation, navigate into the doc folder and run the make command:

```bash
cd doc
sphinx-build -b html -d build/doctrees source build/html -W
```

You can now view the documentation in your favorite browser by opening `build/html/index.html`.
