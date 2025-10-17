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
"""Utils to build queens on remote resource."""

import argparse
import json
import sys

from queens.utils.remote_operations import RemoteConnection

DEFAULT_PACKAGE_MANAGER = "mamba"
FALLBACK_PACKAGE_MANAGER = "conda"
SUPPORTED_PACKAGE_MANAGERS = [DEFAULT_PACKAGE_MANAGER, FALLBACK_PACKAGE_MANAGER]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build queens environment on remote machine.")
    parser.add_argument(
        "--host", type=str, required=True, help="hostname or ip address of remote host"
    )
    parser.add_argument("--user", type=str, default=None, required=False, help="remote username")
    parser.add_argument(
        "--remote-python", type=str, required=True, help="path to python environment on remote host"
    )
    parser.add_argument(
        "--remote-queens-repository",
        type=str,
        required=True,
        help="path to queens repository on remote host",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        required=False,
        default=None,
        help=(
            "gateway connection (proxyjump) for remote connection in json format,"
            ' e.g. \'{"host": "user@host"}\''
        ),
    )
    parser.add_argument(
        "--package-manager",
        type=str,
        default=DEFAULT_PACKAGE_MANAGER,
        choices=SUPPORTED_PACKAGE_MANAGERS,
        help="package manager used for the creation of the remote environment",
    )

    args = parser.parse_args(sys.argv[1:])

    remote_connection = RemoteConnection(
        host=args.host,
        user=args.user,
        remote_python=args.remote_python,
        remote_queens_repository=args.remote_queens_repository,
        gateway=args.gateway if args.gateway is None else json.loads(args.gateway),
    )
    remote_connection.open()
    remote_connection.sync_remote_repository()
    remote_connection.build_remote_environment(package_manager=args.package_manager)
