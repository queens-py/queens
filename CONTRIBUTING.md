# :busts_in_silhouette: Contributing to QUEENS
Thank you very much for your willingness to contribute to QUEENS! We strongly believe in the synergy
effect of developing and using QUEENS together as a community. We welcome all types of
contributions, irrespective of size and complexity.

### Types of contributions
- [Issues](#rotating_light-Issues)
- [Pull requests](#fishing_pole_and_fish-pull-requests)
- [Documentation](#book-documentation)


## :rotating_light: Issues
Issues are generally used to remind or inform yourself or others about certain things in the
software. We use them to report bugs, start a feature request, or plan tasks. In case you have a
general question please refer to [Github Discussions](https://github.com/queens-py/queens/discussions).

To create an issue select one of our templates and provide a detailed description. We use labels
to organize our issues, please label it with the mandatory labels
- `status:` label
- `topic:` label
- `type:` label

More [labels](https://github.com/queens-py/queens/labels) can of course be assigned if they
contribute to categorizing the issue.

Before you open a new issue, please check within the existing issues if your bug has
already been reported. Opening an issue is a valid contribution on its own and does not mean you
have to solve them yourself.


## :fishing_pole_and_fish: Pull requests

### 1. Install QUEENS in developer mode
Install QUEENS as described in the [README.md](README.md) and run:
```
pip install -e .[develop]
```

### 2. Configure our git-hooks
To help you write style-compliant code, we use the [pre-commit](https://pre-commit.com/) package to manage all our git
hooks automatically. Please run:
```
pre-commit install --install-hooks --overwrite
pre-commit install --hook-type commit-msg
```

### 3. Code development

#### Coding style
QUEENS code follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Non-compliant code
will lead to failing CI pipelines and will therefore not be merged.
The code checks are conducted with [pylint](https://pylint.org/),
[isort](https://github.com/PyCQA/isort), and [black](https://github.com/psf/black).
Compliance with [Google style docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
is checked with [pydocstyle](https://github.com/PyCQA/pydocstyle).
Complete and meaningful docstrings are required as they are used to generate the
[documentation](#reading-and-writing-documentation).

#### Commit messages
Please provide meaningful commit messages based on the
[convential commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/).
These are verified by the commit-msg hook (managed by [commitizen](https://github.com/commitizen-tools/commitizen)).

### 4. Test your code
New code must be tested. Please also make sure that all existing tests pass by running `pytest` in
your source directory. For further information, see our [testing README.md](tests/README.md).

### 5. Submit a pull request
Please use the available pull request template and fill out all sections of the template.
When you have submitted a pull request and the CI pipeline passes, it will be reviewed.
Once your pull request is approved, there is a 24h waiting time until the branch is merged into the
main branch by the QUEENS maintainers. This ensures that the community has a chance to have a final look over the changes.


## :book: Documentation
We use [Sphinx](https://www.sphinx-doc.org/en/master/#) to generate the
[QUEENS documentation](https://queens-py.github.io/queens).
We believe that documentation is essential and therefore welcome any improvements.
