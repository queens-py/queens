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
"""Module supplies functions to conduct operation on remote resource."""

import atexit
import json
import logging
import pickle
import socket
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import cloudpickle
from fabric import Connection
from invoke.exceptions import UnexpectedExit

from queens.utils.path import PATH_TO_ROOT, is_empty
from queens.utils.rsync import assemble_rsync_command
from queens.utils.run_subprocess import start_subprocess

_logger = logging.getLogger(__name__)

DEFAULT_PACKAGE_MANAGER = "mamba"
FALLBACK_PACKAGE_MANAGER = "conda"
SUPPORTED_PACKAGE_MANAGERS = [DEFAULT_PACKAGE_MANAGER, FALLBACK_PACKAGE_MANAGER]


class RemoteConnection(Connection):
    """This is class wrapper around the Connection class of fabric.

    Attributes:
        remote_python: Path to Python with installed (editable) QUEENS
            (see remote_queens_repository)
        remote_queens_repository: Path to the QUEENS source code on the remote host
    """

    def __init__(
        self,
        host: str,
        remote_python: str | Path,
        remote_queens_repository: str | Path,
        user: str | None = None,
        gateway: dict | Connection | None = None,
    ):
        """Initialize RemoteConnection object.

        Args:
            host: address of remote host
            remote_python: Path to Python with installed (editable) QUEENS
                            (see remote_queens_repository)
            remote_queens_repository: Path to the QUEENS source code on the remote host
            user: Username on remote machine
            gateway: An object to use as a proxy or gateway for this
                                            connection. See docs of Fabric's Connection object for
                                            details.
        """
        if isinstance(gateway, dict):
            gateway = Connection(**gateway)

        super().__init__(host, user=user, gateway=gateway)
        self.remote_python = remote_python
        _logger.debug("remote python path: %s", self.remote_python)

        self.remote_queens_repository = remote_queens_repository
        _logger.debug("remote queens repository: %s", self.remote_queens_repository)

    def open(self) -> None:
        """Initiate the SSH connection."""
        super().open()
        atexit.register(self.close)

    def start_cluster(
        self,
        workload_manager: str,
        dask_cluster_kwargs: dict,
        dask_cluster_adapt_kwargs: dict,
        experiment_dir: str,
    ) -> tuple[Any, Any]:
        """Start a Dask Cluster remotely using an ssh connection.

        Args:
            workload_manager: Workload manager ("pbs" or "slurm") on cluster
            dask_cluster_kwargs: Collection of keyword arguments to be forwarded to DASK Cluster
            dask_cluster_adapt_kwargs: Collection of keyword arguments to be forwarded to DASK
                Cluster adapt method
            experiment_dir: Directory holding all data of QUEENS experiment on remote
        Returns:
            Return value of function
        """
        if self.client is None:
            raise RuntimeError("Client has not been initialized.")

        _logger.info("Starting Dask cluster on %s", self.host)

        python_cmd = (
            "source /etc/profile;"
            f"{self.remote_python} "
            f"{Path(self.remote_queens_repository) / 'src/queens/utils/start_dask_cluster.py'} "
            f"--workload-manager {workload_manager} "
            f"--dask-cluster-kwargs '{json.dumps(dask_cluster_kwargs)}' "
            f"--dask-cluster-adapt-kwargs '{json.dumps(dask_cluster_adapt_kwargs)}' "
            f"--experiment-dir {experiment_dir}"
        )
        _logger.debug("Starting cluster with command:")
        _logger.debug("%s", python_cmd)
        _, stdout, stderr = self.client.exec_command(python_cmd, get_pty=True)

        return stdout, stderr

    def run_function(
        self, func: Callable, *func_args: Any, wait: bool = True, **func_kwargs: Any
    ) -> Any:
        """Run a python function remotely using an ssh connection.

        Args:
            func: Function that is executed
            func_args: Additional arguments for the functools.partial function
            wait: Flag to decide whether to wait for result of function
            func_kwargs: Additional keyword arguments for the functools.partial function
        Returns:
            Return value of function
        """
        _logger.info("Running %s on %s", func.__name__, self.host)
        func_file_name = f"temp_func_{str(uuid.uuid4())}.pickle"
        output_file_name = f"output_{str(uuid.uuid4())}.pickle"
        python_cmd = (
            f"{self.remote_python} -c 'import pickle; from pathlib import Path;"
            f'file = open("{func_file_name}", "rb");'
            f"func = pickle.load(file); file.close();"
            f'Path("{func_file_name}").unlink(); '
            f"result = func();"
            f'file = open("{output_file_name}", "wb");'
            f"pickle.dump(result, file); file.close();'"
        )
        partial_func = partial(func, *func_args, **func_kwargs)  # insert function arguments
        with open(func_file_name, "wb") as file:
            cloudpickle.dump(partial_func, file)  # pickle function by value

        self.put(func_file_name)  # upload local function file
        Path(func_file_name).unlink()  # delete local function file

        if not wait:
            if self.client is None:
                raise RuntimeError("Client has not been initialized.")
            _, stdout, stderr = self.client.exec_command(python_cmd, get_pty=True)
            return stdout, stderr

        try:
            result = self.run(python_cmd, in_stream=False, hide=True)  # run function remote
        except UnexpectedExit as unexpected_exit:
            _logger.debug(unexpected_exit.result.stdout)
            _logger.debug(unexpected_exit.result.stderr)
            raise unexpected_exit
        _logger.debug(result.stdout)
        _logger.debug(result.stderr)
        self.get(output_file_name)  # download result

        self.run(f"rm {output_file_name}", in_stream=False)  # delete remote files

        with open(output_file_name, "rb") as file:  # read return value from output file
            return_value = pickle.load(file)

        Path(output_file_name).unlink()  # delete local output file

        return return_value

    def get_free_local_port(self) -> int:
        """Get a free port on localhost."""
        return get_port()

    def get_free_remote_port(self) -> Any:
        """Get a free port on remote host."""
        return self.run_function(get_port)

    def open_port_forwarding(
        self, local_port: int | None = None, remote_port: int | None = None
    ) -> tuple[int, Any]:
        """Open port forwarding.

        Args:
            local_port: Free local port
            remote_port: Free remote port
        Returns:
            Used local port
            Used remote port
        """
        if local_port is None:
            local_port = self.get_free_local_port()
        if remote_port is None:
            remote_port = self.get_free_remote_port()

        proxyjump = ""
        if self.gateway is not None:
            proxyjump = f"-J {self.gateway.user}@{self.gateway.host}:{self.gateway.port}"
        cmd = (
            f"ssh {proxyjump} -f -N -L {local_port}:{self.host}:{remote_port} "
            f"{self.user}@{self.host}"
        )
        _logger.debug("\nOpening port-forwarding '%s'\n", cmd)

        start_subprocess(cmd)
        _logger.debug("Port-forwarding opened successfully.")

        kill_cmd = f'pkill -f "{cmd}"'
        atexit.register(start_subprocess, kill_cmd)

        return local_port, remote_port

    def create_remote_directory(self, remote_directory: str | Path) -> None:
        """Make a directory (including parents) on the remote host.

        Args:
            remote_directory: Path of the directory that will be created
        """
        _logger.debug("Creating folder %s on %s@%s.", remote_directory, self.user, self.host)
        result = self.run(f"mkdir -v -p {remote_directory}", in_stream=False)
        stdout = result.stdout
        if stdout:
            _logger.debug(stdout)
        else:
            _logger.debug("%s already exists on %s@%s.", remote_directory, self.user, self.host)

    def sync_remote_repository(self) -> None:
        """Synchronize local and remote QUEENS source files."""
        _logger.info("Syncing remote QUEENS repository with local one...")
        start_time = time.time()
        self.create_remote_directory(self.remote_queens_repository)

        source = f"{PATH_TO_ROOT}/"
        self.copy_to_remote(
            source, self.remote_queens_repository, exclude=".git", filters=":- .gitignore"
        )
        _logger.info("Sync of remote repository was successful.")
        _logger.info("It took: %s s.\n", time.time() - start_time)

    def copy_to_remote(
        self,
        source: str | Path | Sequence,
        destination: Path | str,
        verbose: bool = True,
        exclude: str | Sequence | None = None,
        filters: str | None = None,
    ) -> None:
        """Copy files or folders to remote.

        Args:
            source: Paths to copy
            destination: Destination relative to host
            verbose: True for verbose
            exclude: Options to exclude
            filters: Filters for rsync
        """
        if not is_empty(source):
            host = f"{self.user}@{self.host}"
            _logger.debug("Copying from %s to %s", source, destination)
            remote_shell_command = None
            if self.gateway is not None:
                remote_shell_command = f"ssh {self.gateway.user}@{self.gateway.host} ssh"
                _logger.debug("Using remote shell command %s", remote_shell_command)
            rsync_cmd = assemble_rsync_command(
                source,
                destination,
                verbose=verbose,
                archive=True,
                exclude=exclude,
                filters=filters,
                rsh=remote_shell_command,
                host=host,
                rsync_options=["--out-format='%n'", "--checksum"],
            )
            # Run rsync command
            result = self.local(rsync_cmd, in_stream=False)
            _logger.debug(result.stdout)
            _logger.debug("Copying complete.")
        else:
            _logger.debug("List of source files was empty. Did not copy anything.")

    def build_remote_environment(
        self,
        package_manager: str = DEFAULT_PACKAGE_MANAGER,
    ) -> None:
        """Build remote QUEENS environment.

        Args:
            package_manager: Package manager used for the creation of the environment ("mamba" or
                "conda")
        """
        if package_manager not in SUPPORTED_PACKAGE_MANAGERS:
            raise ValueError(
                f"The package manager '{package_manager}' is not supported.\n"
                f"Supported package managers are: {SUPPORTED_PACKAGE_MANAGERS}"
            )
        remote_connect = f"{self.user}@{self.host}"

        # check if requested package_manager is installed on remote machine:
        def package_manager_exists_remote(package_manager_name: str) -> bool:
            """Check if requested package manager exists on remote.

            Args:
                package_manager_name: name of package manager
            """
            result_which = self.run(f"which {package_manager_name}")
            if result_which.stderr:
                message = (
                    f"Could not find requested package manager '{package_manager_name}' "
                    f"on '{remote_connect}'."
                )
                if package_manager_name == DEFAULT_PACKAGE_MANAGER:
                    _logger.warning(message)
                    _logger.warning(
                        "Trying to fall back to the '%s' package manager.", FALLBACK_PACKAGE_MANAGER
                    )
                    package_manager_exists_remote(package_manager_name=FALLBACK_PACKAGE_MANAGER)
                else:
                    raise RuntimeError(message)
                return False
            return True

        if not package_manager_exists_remote(package_manager_name=package_manager):
            package_manager = FALLBACK_PACKAGE_MANAGER

        _logger.info("Build remote QUEENS environment...")
        start_time = time.time()
        environment_name = Path(self.remote_python).parents[1].name
        command_string = (
            f"cd {self.remote_queens_repository}; "
            f"{package_manager} remove --name {environment_name} --all -y;"
            f"{package_manager} env create -f environment.yml --name {environment_name}; "
            f"{package_manager} activate {environment_name};"
            f"pip install -e ."
        )
        result = self.run(command_string, in_stream=False)

        _logger.debug(result.stdout)
        _logger.info("Build of remote queens environment was successful.")
        _logger.info("It took: %s s.\n", time.time() - start_time)


def get_port() -> int:
    """Get free port.

    Returns:
        Free port
    """
    sock = socket.socket()
    sock.bind(("", 0))
    return int(sock.getsockname()[1])


VALID_CONNECTION_TYPES = {"remote_connection": RemoteConnection}
