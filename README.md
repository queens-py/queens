<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="doc/source/images/queens_logo_night.svg">
  <source media="(prefers-color-scheme: light)" srcset="doc/source/images/queens_logo_day.svg">
  <img alt="QUEENS logo" src="doc/source/images/queens_logo_night.svg" width="300">
</picture>

</div>

<div align="center">

[![precommit](./doc/source/images/precommit_badge.svg)](https://github.com/pre-commit/pre-commit)
[![black](./doc/source/images/black_badge.svg)](https://github.com/psf/black)
[![pylint](./doc/source/images/pylint_badge.svg)](https://github.com/pylint-dev/pylint)

</div>

<div align="center">

[![tests-local-main](https://github.com/queens-py/queens/actions/workflows/tests_local.yml/badge.svg?branch=main)](https://github.com/queens-py/queens/actions/workflows/tests_local.yml?query=branch:main)
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
>**Prerequisites**: Unix system and environment management system (we recommend [miniforge](https://conda-forge.org/download/))
<!---prerequisites marker, do not remove this comment-->

<!---installation marker, do not remove this comment-->
Clone the QUEENS repository to your local machine. Navigate to its base directory, then:
```
conda env create
conda activate queens
pip install -e .
```
<!---installation marker, do not remove this comment-->

## :crown: Workflow example

Let's consider a parallelized Monte Carlo simulation of the [Ishigami function](https://www.sfu.ca/~ssurjano/ishigami.html):
<!---example marker, do not remove this comment-->
```python
from queens.distributions import BetaDistribution, NormalDistribution, UniformDistribution
from queens.drivers import FunctionDriver
from queens.global_settings import GlobalSettings
from queens.iterators import MonteCarloIterator
from queens.main import run_iterator
from queens.models import SimulationModel
from queens.parameters import Parameters
from queens.schedulers import LocalScheduler

if __name__ == "__main__":
    # Set up the global settings
    global_settings = GlobalSettings(experiment_name="monte_carlo_uq", output_dir=".")

    # Set up the uncertain parameters
    x1 = UniformDistribution(lower_bound=-3.14, upper_bound=3.14)
    x2 = NormalDistribution(mean=0.0, covariance=1.0)
    x3 = BetaDistribution(lower_bound=-3.14, upper_bound=3.14, a=2.0, b=5.0)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Set up the model
    driver = FunctionDriver(parameters=parameters, function="ishigami90")
    scheduler = LocalScheduler(
        experiment_name=global_settings.experiment_name, num_jobs=2, num_procs=4
    )
    model = SimulationModel(scheduler=scheduler, driver=driver)

    # Set up the algorithm
    iterator = MonteCarloIterator(
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

Your contributions are welcome! Please follow our [contributing guidelines](https://github.com/queens-py/queens/blob/main/CONTRIBUTING.md) and [code of conduct](https://github.com/queens-py/queens/blob/main/CODE_OF_CONDUCT.md).

## :page_with_curl: How to cite
If you use QUEENS in your work, please cite the relevant method papers and

<!---citation marker, do not remove this comment-->
```bib
@misc{queens,
  author       = {QUEENS},
  title        = {QUEENS: An Open-Source Python Framework for Solver-Independent Analyses of Large-Scale Computational Models},
  year         = {2025},
  howpublished = {\url{https://www.queens-py.org}}
}
```
<!---citation marker, do not remove this comment-->

## :woman_judge: License
<!---license marker, do not remove this comment-->
Licensed under GNU LGPL-3.0 (or later). See [LICENSE](LICENSE).
<!---license marker, do not remove this comment-->
