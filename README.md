<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="doc/source/images/queens_logo_night.svg">
  <source media="(prefers-color-scheme: light)" srcset="doc/source/images/queens_logo_day.svg">
  <img alt="QUEENS logo" src="doc/source/images/queens_logo_night.svg">
</picture>

</div>

<br>

<div align="center">

[![precommit](./doc/source/images/precommit_badge.svg)](https://github.com/pre-commit/pre-commit)
[![black](./doc/source/images/black_badge.svg)](https://github.com/psf/black)
[![pylint](./doc/source/images/pylint_badge.svg)](https://github.com/pylint-dev/pylint)

</div>

<div align="center">

[![tests-local-main](https://github.com/queens-py/queens/actions/workflows/local_testsuite.yml/badge.svg?branch=main)](https://github.com/queens-py/queens/actions/workflows/local_testsuite.yml?query=branch:main)
[![build-documentation-main](https://github.com/queens-py/queens/actions/workflows/build_documentation.yml/badge.svg?branch=main)](https://github.com/queens-py/queens/actions/workflows/build_documentation.yml?query=branch:main)

</div>

<!---description marker, do not remove this comment-->
QUEENS (**Q**uantification of **U**ncertain **E**ffects in **En**gineering **S**ystems) is a Python framework for solver-independent multi-query analyses of large-scale computational models.
<!---description marker, do not remove this comment-->

<!---capabilities marker, do not remove this comment-->
:chart_with_upwards_trend: **QUEENS** offers a large collection of cutting-edge algorithms for deterministic and probabilistic analyses such as:
* parameter studies and identification
* sensitivity analysis
* surrogate modeling
* uncertainty quantification
* Bayesian inverse analysis

:fairy_man: **QUEENS** provides a modular architecture for:
* parallel queries of large-scale computational models
* robust data, resource, and error management
* easy switching between analysis types
* smooth scaling from laptop to HPC cluster
<!---capabilities marker, do not remove this comment-->

:globe_with_meridians: **Website**: [queens-py.org](https://www.queens-py.org)

:book: **Documentation**: [queens-py.github.io/queens](https://queens-py.github.io/queens)

## :rocket: Getting started

<!---prerequisites marker, do not remove this comment-->
>**Prerequisites**: Python 3.12 or newer. For development, use [Pixi](https://pixi.sh/latest/).
<!---prerequisites marker, do not remove this comment-->
### Easy installation
<!---installation marker, do not remove this comment-->
Clone the QUEENS repository and install it from the source checkout with:
```bash
pip install .
```
We recommend using some form of environment management instead of installing into your system Python.
For more details, see [the QUEENS documentation](https://queens-py.github.io/queens/introduction.html#installation).

Optional runtime extras can be installed with:
```bash
pip install ".[tutorials]"
pip install ".[fourc]"
pip install ".[all]"
```

### Recommended installation
We recommend a modern project-based workflow based on [Pixi](https://pixi.sh/latest/) especially for development.
After cloning the repository, installing with Pixi is as easy as:
```bash
pixi install
pixi run install-editable
```
The default Pixi environment contains the core QUEENS dependencies. Use `all` for runtime
extras without development tools. For named Pixi environments, run
`pixi run -e <environment> install-editable` once for each environment you want to use.

### Development installation
For development, we recommend using [Pixi](https://pixi.sh/latest/) together with the `dev` environment.
`dev` contains the full contributor setup, including development tools, tutorials, and the
4C interface dependencies.
Clone the repository and install with:
```bash
pixi install --environment dev
pixi run -e dev install-editable
```
Useful development commands then look like:
```bash
pixi run -e dev pytest
pixi run -e dev pre-commit run --all-files
```
<!---installation marker, do not remove this comment-->

## :crown: Workflow example

Let's consider a parallelized Monte Carlo simulation of the [Ishigami function](https://www.sfu.ca/~ssurjano/ishigami.html):
<!---example marker, do not remove this comment-->
```python
from queens.distributions import Beta, Normal, Uniform
from queens.drivers import Function
from queens.global_settings import GlobalSettings
from queens.iterators import MonteCarlo
from queens.main import run_iterator
from queens.models import Simulation
from queens.parameters import Parameters
from queens.schedulers import Local

if __name__ == "__main__":
    # Set up the global settings
    global_settings = GlobalSettings(experiment_name="monte_carlo_uq", output_dir=".")

    with global_settings:
        # Set up the uncertain parameters
        x1 = Uniform(lower_bound=-3.14, upper_bound=3.14)
        x2 = Normal(mean=0.0, covariance=1.0)
        x3 = Beta(lower_bound=-3.14, upper_bound=3.14, a=2.0, b=5.0)
        parameters = Parameters(x1=x1, x2=x2, x3=x3)

        # Set up the model
        driver = Function(parameters=parameters, function="ishigami90")
        scheduler = Local(
            experiment_name=global_settings.experiment_name, num_jobs=2, num_procs=4
        )
        model = Simulation(scheduler=scheduler, driver=driver)

        # Set up the algorithm
        iterator = MonteCarlo(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            seed=42,
            num_samples=1000,
            result_description={"write_results": True, "plot_results": True},
        )

        # Start QUEENS run
        run_iterator(iterator, global_settings=global_settings)
```
<!---example marker, do not remove this comment-->

<div align="center">
<img src="doc/source/images/monte_carlo_uq.png" alt="QUEENS example" width="500"/>
</div>

## :busts_in_silhouette: Contributing

QUEENS is powered by a [community of contributors](https://www.queens-py.org/community/).

Join us—your contributions are welcome! Please follow our [contributing guidelines](https://github.com/queens-py/queens/blob/main/CONTRIBUTING.md) and [code of conduct](https://github.com/queens-py/queens/blob/main/CODE_OF_CONDUCT.md).

## :page_with_curl: How to cite

If you use QUEENS in your work, please cite the relevant method papers and the following article

<!---citation marker, do not remove this comment-->
```bib
@misc{queens,
      title={QUEENS: An Open-Source Python Framework for Solver-Independent Analyses of Large-Scale Computational Models},
      author={Jonas Biehler and Jonas Nitzler and Sebastian Brandstaeter and Maximilian Dinkel and Volker Gravemeier and Lea J. Haeusel and Gil Robalo Rei and Harald Willmann and Barbara Wirthl and Wolfgang A. Wall},
      year={2025},
      eprint={2508.16316},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2508.16316},
      note = {Jonas Biehler, Jonas Nitzler, and Sebastian Brandstaeter contributed equally.}
}
```
<!---citation marker, do not remove this comment-->

## :woman_judge: License
<!---license marker, do not remove this comment-->
Licensed under GNU LGPL-3.0 (or later). See [LICENSE](LICENSE).
<!---license marker, do not remove this comment-->
