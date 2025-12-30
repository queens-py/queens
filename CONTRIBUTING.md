# :busts_in_silhouette: Contributing to QUEENS
Thank you very much for your willingness to contribute to QUEENS! We strongly believe in the synergy effect of developing and using QUEENS together as a community.

We invite you to share your methodological contributions to both deterministic and probabilistic models and analyses, such as:
- Parameter studies and identification
- Sensitivity analysis
- Surrogate modeling
- Uncertainty quantification
- Bayesian inverse analysis
- Optimization

In addition to methodological contributions, we also greatly appreciate infrastructure contributions that help ensure QUEENS runs smoothly and efficiently. These include, but are not limited to, bug fixes and improvements related to
- Code quality
- Performance
- User interface
- Benchmarking
- [Testing](tests/README.md)
- [Tutorials](https://queens-py.github.io/queens/tutorials.html)
- [Documentation](doc/README.md)

We welcome all types of code contributions, irrespective of size and complexity.

> Note: If you're unsure whether your contribution fits within the QUEENS framework, don't hesitate to ask the community by starting a [discussion](https://github.com/queens-py/queens/discussions) or by opening an [issue](https://github.com/queens-py/queens/issues) :blush:

## Contributing on GitHub
### :rotating_light: Issues
Issues are generally used to remind or inform yourself or others about certain things in the
software. We use them to report bugs, start a feature request, or plan tasks. In case you have a
general question, please refer to [GitHub Discussions](https://github.com/queens-py/queens/discussions).

To create an issue, select one of our templates and provide a detailed description. We use labels
to organize our issues, so please label issues with the mandatory labels
- `status:` label
- `topic:` label
- `type:` label

More [labels](https://github.com/queens-py/queens/labels) can of course be assigned if they
contribute to categorizing the issue.

Before you open a new issue, please check within the existing issues if your bug has
already been reported. Opening an issue is a valid contribution on its own and does not mean you
have to solve them yourself.


### :fishing_pole_and_fish: Pull requests

#### 1. Install QUEENS in developer mode
Install QUEENS as described in the [README.md](README.md) and run:
<!---installation_develop marker, do not remove this comment-->
```
pip install -e .[develop]
```
or to do a safe develop install use:
```
pip install -e .[safe_develop]
```
<!---installation_develop marker, do not remove this comment-->

#### 2. Configure our git-hooks
To help you write style-compliant code, we use the [pre-commit](https://pre-commit.com/) package to manage all our git
hooks automatically. Please run:
```
pre-commit install --install-hooks --overwrite
pre-commit install --hook-type commit-msg
```

#### 3. Code development

##### Coding style
QUEENS code follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Non-compliant code
will lead to failing CI pipelines and will therefore not be merged.
The code checks are conducted with [Pylint](https://pylint.org/),
[isort](https://github.com/PyCQA/isort), and [Black](https://github.com/psf/black).
Compliance with [Google style docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
is checked with [ruff](https://github.com/astral-sh/ruff).
Complete and meaningful docstrings are required as they are used to generate the
[documentation](https://www.github.com/queens-py/queens/blob/main/doc/README.md).

##### QUEENS coding conventions
Like every codebase, QUEENS follows some project-specific coding conventions. Below is a list of common ones:
- Use `pathlib.Path` objects instead of strings to handle paths and directories.
- If relative paths within the QUEENS source are needed, use the [relative_path_from_source](queens/utils/path.py#L23) function.
- Decorate the init method of QUEENS objects with the `log_init_args` decorator from [queens/utils/logger_setting.py](queens/utils/logger_settings.py#L239). This automatically logs the arguments passed to the init.
- We only allow disabling pylint warnings for specific lines, not for entire files. If you disable warnings, please use the long pylint description, not just the code.
##### Commit messages
Please provide meaningful commit messages based on the
[Conventional Commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/).
These are verified by the commit-msg hook (managed by [commitizen](https://github.com/commitizen-tools/commitizen)).

#### 4. Test your code
New code must be tested. Please also make sure that all existing tests pass by running `pytest` in
your source directory. For further information, see our [testing README.md](tests/README.md).

#### 5. Submit a pull request
Please use the available pull request template and fill out all sections of the template.
When you have submitted a pull request and the CI pipeline passes, it will be reviewed.
Once your pull request is approved, there is a 24h waiting time (business days only) until the branch is merged into the
main branch by the QUEENS maintainers. This ensures that the community has a chance to have a final look over the changes.
This rule can be circumvented if and only if:
- All the active maintainers approve the pull request.
- The pull request is labeled as a quickfix by one of the maintainers and approved. Examples for this are one-liners, typos or urgent fixes that are time-critical.
