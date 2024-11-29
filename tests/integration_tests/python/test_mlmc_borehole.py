"""Integration test for the Multilevel Monte Carlo iterator.

This test is based on the low-fidelity Borehole function.
"""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.mlmc_iterator import MLMCIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


def test_mlmc_borehole(global_settings):
    """Test case for Monte Carlo iterator.

    Testing both normal num_samples and optimal num_samples.
    """
    # Parameters
    rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
    r = UniformDistribution(lower_bound=100, upper_bound=50000)
    tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
    hu = UniformDistribution(lower_bound=990, upper_bound=1110)
    tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
    hl = UniformDistribution(lower_bound=700, upper_bound=820)
    l = UniformDistribution(lower_bound=1120, upper_bound=1680)
    kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Set up scheduler
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)

    # Set up drivers
    driver0 = FunctionDriver(parameters=parameters, function="borehole83_lofi")
    driver1 = FunctionDriver(parameters=parameters, function="borehole83_hifi")

    # Set up models
    model0 = SimulationModel(scheduler=scheduler, driver=driver0)
    model1 = SimulationModel(scheduler=scheduler, driver=driver1)

    # Set up iterator
    iterator = MLMCIterator(
        seed=42,
        num_samples=[1000, 100],
        result_description={"write_results": True, "plot_results": False},
        models=[model0, model1],
        parameters=parameters,
        global_settings=global_settings,
    )

    # Analysis
    run_iterator(iterator=iterator, global_settings=global_settings)

    result = load_result(path_to_result_file=global_settings.result_file(".pickle"))

    assert result["mean"] == pytest.approx(76.52224796054254)
    assert result["std"] == pytest.approx(1.4321771079107875)

    # Optimal number of samples test

    iterator_optimal = MLMCIterator(
        seed=42,
        num_samples=[1000, 100],
        result_description={"write_results": True, "plot_results": False},
        models=[model0, model1],
        parameters=parameters,
        global_settings=global_settings,
        use_optimal_num_samples=True,
        cost_models=[1, 1000],
        bootstrap_samples=200,
    )

    run_iterator(iterator=iterator_optimal, global_settings=global_settings)
    result = load_result(path_to_result_file=global_settings.result_file(".pickle"))

    assert result["mean"] == pytest.approx(77.9589082063506)
    assert result["std"] == pytest.approx(0.9411428987983472)
