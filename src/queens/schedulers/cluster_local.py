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
"""Cluster scheduler for QUEENS runs."""

import logging
import time

from dask.distributed import Client

from queens.schedulers._cluster_base import (
    VALID_WORKLOAD_MANAGERS,
    _BaseCluster,
    _initialize_dask_cluster,
)
from queens.schedulers._scheduler import Scheduler
from queens.utils.ports import get_port
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


class ClusterLocal(_BaseCluster):
    """Cluster scheduler with local access (no network connection)."""

    def _get_experiment_dir(
        self, experiment_name, experiment_base_dir, overwrite_existing_experiment
    ):
        """Get local experiment directory."""
        return Scheduler.local_experiment_dir(
            self, experiment_name, experiment_base_dir, overwrite_existing_experiment
        )

    def _start_cluster(self, dask_cluster_kwargs, dask_cluster_adapt_kwargs):
        """Start a Dask cluster and connect a client locally."""
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, self.workload_manager)
        dask_cluster_cls = dask_cluster_options["dask_cluster_cls"]

        port = get_port()
        port_dashboard = get_port()

        scheduler_options = {
            "port": port,
            "dashboard_address": port_dashboard,
            "allowed_failures": self.allowed_failures,
        }
        if self.cluster_internal_address:
            scheduler_options["contact_address"] = f"{self.cluster_internal_address}:{port}"

        dask_cluster_kwargs["scheduler_options"] = scheduler_options

        try:
            cluster = _initialize_dask_cluster(  # pylint: disable=duplicate-code
                _logger,
                dask_cluster_cls,
                dask_cluster_kwargs,
                dask_cluster_adapt_kwargs,
                self.experiment_dir,
            )
        except Exception as exc:
            raise RuntimeError("Failed to initialize local Dask cluster") from exc

        for i in range(20, 0, -1):  # 20 tries to connect
            _logger.debug("Trying to connect to Dask Cluster: try #%d", i)
            try:
                client = Client(cluster)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError("Failed to connect to local Dask cluster") from exc
                time.sleep(1)

        return client, port_dashboard

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory."""
        return Scheduler.copy_files_to_experiment_dir(self, paths)
