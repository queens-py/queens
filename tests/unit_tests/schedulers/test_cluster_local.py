#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Unit tests for the ClusterLocal scheduler."""

import itertools

import pytest

from queens.schedulers import _cluster_base, cluster_local
from queens.schedulers._cluster_base import VALID_WORKLOAD_MANAGERS, _BaseCluster
from queens.schedulers._dask import Dask
from queens.schedulers.cluster_local import ClusterLocal


@pytest.fixture(name="mock_dask_layer")
def fixture_mock_dask_layer(monkeypatch):
    """Replace the real dask-jobqueue layer with lightweight fakes."""
    created = {}

    class FakeCluster:
        """Stand-in for ``SLURMCluster``/``PBSCluster``."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.adapt_kwargs = None
            created["cluster"] = self

        def adapt(self, **kwargs):
            """Record the requested adapt kwargs."""
            self.adapt_kwargs = kwargs

        def job_script(self):
            """Return a dummy jobscript."""
            return "#!/bin/bash\n# fake dask jobscript\n"

    class FakeFuture:
        """Stand-in for a Dask future."""

        @staticmethod
        def result(timeout=None):  # pylint: disable=unused-argument
            """Return a dummy result immediately."""
            return "Dummy job"

    class FakeClient:
        """Stand-in for ``dask.distributed.Client``."""

        def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
            created["client"] = self

        @staticmethod
        def submit(function, *args, **kwargs):  # pylint: disable=unused-argument
            """Return a fake future without running anything."""
            return FakeFuture()

    # deterministic, unique ports instead of querying the operating system
    ports = itertools.count(10000)
    monkeypatch.setattr(cluster_local, "get_port", lambda: next(ports))
    monkeypatch.setattr(cluster_local, "Client", FakeClient)

    # use the real directive logic but swap in the fake cluster class; fresh
    # copies of the skip lists keep the tests independent of one another
    fake_managers = {
        name: {
            "dask_cluster_cls": FakeCluster,
            "job_extra_directives": options["job_extra_directives"],
            "job_directives_skip": list(options["job_directives_skip"]),
        }
        for name, options in VALID_WORKLOAD_MANAGERS.items()
    }
    monkeypatch.setattr(_cluster_base, "VALID_WORKLOAD_MANAGERS", fake_managers)
    monkeypatch.setattr(cluster_local, "VALID_WORKLOAD_MANAGERS", fake_managers)

    return created


def build_cluster_local(test_name, tmp_path, **overrides):
    """Construct a ``ClusterLocal`` scheduler with sensible test defaults."""
    kwargs = {
        "experiment_name": test_name,
        "workload_manager": "slurm",
        "walltime": "00:10:00",
        "num_jobs": 3,
        "min_jobs": 1,
        "num_procs": 1,
        "num_nodes": 1,
        "experiment_base_dir": tmp_path,
        "overwrite_existing_experiment": True,
    }
    kwargs.update(overrides)
    return ClusterLocal(**kwargs)


def test_clusterlocal_is_a_cluster_scheduler():
    """ClusterLocal should be a cluster scheduler built on the Dask base."""
    assert issubclass(ClusterLocal, _BaseCluster)
    assert issubclass(ClusterLocal, Dask)


def test_dask_cluster_kwargs_slurm(mock_dask_layer, test_name, tmp_path):
    """The assembled SLURM cluster arguments should match the configuration."""
    build_cluster_local(test_name, tmp_path)
    cluster = mock_dask_layer["cluster"]

    assert cluster.kwargs["job_name"] == test_name
    assert cluster.kwargs["memory"] == "10TB"
    # the walltime is increased by 5 minutes to let workers shut down in time
    assert cluster.kwargs["walltime"] == "00:15:00"
    assert cluster.kwargs["job_extra_directives"] == ["--ntasks=1"]
    assert cluster.kwargs["worker_extra_args"] == [
        "--lifetime",
        "720s",
        "--lifetime-stagger",
        "2m",
    ]
    # one core/process per worker - parallelism is over jobs, not threads
    assert cluster.kwargs["cores"] == 1
    assert cluster.kwargs["processes"] == 1
    assert cluster.kwargs["n_workers"] == 1


def test_adapt_kwargs_use_job_limits(mock_dask_layer, test_name, tmp_path):
    """The Dask cluster should adapt between ``min_jobs`` and ``num_jobs``."""
    build_cluster_local(test_name, tmp_path, num_jobs=4, min_jobs=2)
    cluster = mock_dask_layer["cluster"]
    assert cluster.adapt_kwargs == {"minimum_jobs": 2, "maximum_jobs": 4}


def test_ntasks_uses_nodes_and_procs(mock_dask_layer, test_name, tmp_path):
    """``--ntasks`` should be the product of nodes and processors per node."""
    build_cluster_local(test_name, tmp_path, num_nodes=2, num_procs=3)
    cluster = mock_dask_layer["cluster"]
    assert cluster.kwargs["job_extra_directives"] == ["--ntasks=6"]


def test_scheduler_options_without_internal_address(mock_dask_layer, test_name, tmp_path):
    """Without an internal address no ``contact_address`` should be set."""
    build_cluster_local(test_name, tmp_path)
    scheduler_options = mock_dask_layer["cluster"].kwargs["scheduler_options"]
    assert scheduler_options["allowed_failures"] == 5
    assert "contact_address" not in scheduler_options


def test_scheduler_options_with_internal_address(mock_dask_layer, test_name, tmp_path):
    """An internal address should be turned into a ``contact_address``."""
    build_cluster_local(test_name, tmp_path, cluster_internal_address="10.0.0.1")
    scheduler_options = mock_dask_layer["cluster"].kwargs["scheduler_options"]
    port = scheduler_options["port"]
    assert scheduler_options["contact_address"] == f"10.0.0.1:{port}"


@pytest.mark.usefixtures("mock_dask_layer")
def test_jobscript_is_written(test_name, tmp_path):
    """The dask jobscript should be written into the experiment directory."""
    scheduler = build_cluster_local(test_name, tmp_path)
    jobscript = scheduler.experiment_dir / "dask_jobscript.sh"
    assert jobscript.exists()
    assert "fake dask jobscript" in jobscript.read_text()


def test_pbs_workload_manager(mock_dask_layer, test_name, tmp_path):
    """The PBS workload manager should produce PBS node directives."""
    build_cluster_local(test_name, tmp_path, workload_manager="pbs")
    cluster = mock_dask_layer["cluster"]
    assert cluster.kwargs["job_extra_directives"] == ["-l nodes=1:ppn=1"]


def test_skip_directives_not_duplicated_across_schedulers(mock_dask_layer, test_name, tmp_path):
    """Creating several schedulers must not accumulate skip directives."""
    build_cluster_local(test_name, tmp_path, queue=None)
    build_cluster_local(test_name, tmp_path, queue=None)
    skip_directives = mock_dask_layer["cluster"].kwargs["job_directives_skip"]
    assert skip_directives.count("#SBATCH -p") == 1
