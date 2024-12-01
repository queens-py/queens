"""Test for Control Varaites Iterator."""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.control_variates_iterator import ControlVariatesIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


def test_control_variates(global_settings):
    """Test function for control variates."""
    n0 = 100

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

    iterator = ControlVariatesIterator(
        models=[model1, model0],
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cvs=10 * n0,
        use_optn=False,
    )

    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    assert res["mean"] == pytest.approx(77.03460846952085)
    assert res["std"] == pytest.approx(1.3774480043137558)
