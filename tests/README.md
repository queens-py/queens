# :guardswoman: Testing

> **The golden rule of testing:**
> If it is not tested, it does not work.

Therefore, we test the QUEENS code base

- to ensure our code is working as expected
- to check compatibility w.r.t. to new features
- to find bugs faster

## :construction_worker: Writing tests
- New tests are required if a new feature is introduced (see our [contributing guidelines](https://github.com/queens-py/queens/blob/main/CONTRIBUTING.md)).
- Our tests are written according to the [arrange-act-assert](https://docs.pytest.org/en/stable/explanation/anatomy.html) principle.
- Whenever possible, use [pytest fixtures](https://docs.pytest.org/en/latest/explanation/fixtures.html) to parameterize tests.

## :running_woman: Running tests
QUEENS is tested using [pytest](https://docs.pytest.org/en/stable/index.html). For a comprehensive list of pytest commands, see [here](https://docs.pytest.org/en/stable/how-to/usage.html). Some additional useful commands to test QUEENS are listed in the following:

| Test                          | Command                                       |
| ----------------------------- | --------------------------------------------- |
| In parallel with pytest-xdist | `pytest -n <num_workers>`                     |
| With verbose output           | `pytest -ra -v`                               |
| With logging output           | `pytest -o log_cli=true --log-cli-level=INFO` |
| With coverage report          | `pytest --cov-report=html --cov`              |
| Only the last failed          | `pytest --lf`                                 |

### :bookmark: Pytest markers
In QUEENS, tests are organized using pytest markers. This allows you to run all tests in a group with a single command:

| Description                     | Command                             |
| ------------------------------- | ----------------------------------- |
| Unit tests                      | `pytest -m unit_tests`              |
| Integration tests               | `pytest -m integration_tests`       |
| 4C integration test (see below) | `pytest -m integration_tests_fourc` |
| List markers                    | `pytest --markers`                  |

### :four_leaf_clover: Integration tests with 4C
For the integration tests in QUEENS that require the multiphysics simulation framework [4C](https://github.com/4C-multiphysics/4C), the user needs to create a **symbolic link** to the 4C-executable and store it under `<queens-base-dir>/config`:
```
ln -s <path-to-4C-build-directory> <queens-base-dir>/config/4C_build
```
