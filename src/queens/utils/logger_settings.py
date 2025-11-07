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
"""Logging in QUEENS."""

import functools
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, ParamSpec

from queens.utils.printing import get_str_table

LIBRARY_LOGGER_NAME = "queens"


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL.

    Attributes:
        level: Logging level
    """

    def __init__(self, level: int) -> None:
        """Initiatlize the logging filter.

        Args:
            level: Logging level
        """
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the logging record.

        Args:
            record: Logging record object

        Returns:
            Filter logging record
        """
        return record.levelno <= self.level


class NewLineFormatter(logging.Formatter):
    """Formatter splitting multiline messages into single line messages.

    A logged message that consists of more than one line - contains a new line char - is split
    into multiple single line messages that all have the same format. Without this the overall
    format of the logging is broken for multiline messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Override format function.

        Args:
            record: Logging record object
        Returns:
            Logged message in supplied format split into single lines
        """
        formatted_message = super().format(record)

        if record.message != "":
            parts = formatted_message.split(record.message)
            formatted_message = formatted_message.replace("\n", "\n" + parts[0])

        return formatted_message


def setup_logger(
    logger: logging.Logger = logging.getLogger(LIBRARY_LOGGER_NAME), debug: bool = False
) -> logging.Logger:
    """Set up the main QUEENS logger.

    Args:
        logger: Logger instance that should be set up
        debug: Indicates debug mode and controls level of logging

    Returns:
        QUEENS logger object
    """
    # The default logging level is INFO (for QUEENS)
    # If the parent logger uses a lower level (e.g. pytest) that level is set
    if logger.parent is not None:
        parent_level = logger.parent.getEffectiveLevel()
        logger.setLevel(min(parent_level, logging.INFO))

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        # deactivate logging for specific modules
        logging.getLogger("arviz").setLevel(logging.CRITICAL)
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
        logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
        logging.getLogger("numba").setLevel(logging.CRITICAL)

    return logger


def setup_stream_handler(logger: logging.Logger) -> None:
    """Set up a stream handler.

    Args:
        logger: Logger object to add the stream handler to
    """
    # a plain, minimal formatter for streamhandlers
    stream_formatter = NewLineFormatter("%(message)s")

    # set up logging to stdout
    console_stdout = logging.StreamHandler(stream=sys.stdout)
    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(logger.level)
    console_stdout.setFormatter(stream_formatter)
    logger.addHandler(console_stdout)

    # set up logging to stderr
    console_stderr = logging.StreamHandler(stream=sys.stderr)
    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(logger.level, logging.ERROR))
    console_stderr.setFormatter(stream_formatter)

    logger.addHandler(console_stderr)


def setup_file_handler(logger: logging.Logger, log_file_path: Path) -> None:
    """Set up a file handler.

    Args:
        logger: Logger object to add the stream handler to
        log_file_path: Path of the logging file
    """
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_formatter = NewLineFormatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)


def setup_basic_logging(
    log_file_path: Path,
    logger: logging.Logger = logging.getLogger(LIBRARY_LOGGER_NAME),
    debug: bool = False,
) -> None:
    """Setup basic logging.

    Args:
        log_file_path: Path to the log-file
        logger: Logger instance that should be set up
        debug: Indicates debug mode and controls level of logging
    """
    logger = setup_logger(logger, debug)
    setup_stream_handler(logger)
    setup_file_handler(logger, log_file_path)


def setup_cli_logging(debug: bool = False) -> None:
    """Set up logging for CLI utils.

    Args:
        debug: Indicates debug mode and controls level of logging
    """
    library_logger = setup_logger(debug=debug)
    setup_stream_handler(library_logger)


def setup_cluster_logging() -> None:
    """Setup cluster logging."""
    level_min = logging.INFO

    logging.basicConfig(
        level=level_min,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
    )

    console_stdout = logging.StreamHandler(stream=sys.stdout)
    console_stderr = logging.StreamHandler(stream=sys.stderr)

    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(level_min)

    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(level_min, logging.ERROR))

    # set a format which is simpler for console use
    formatter = NewLineFormatter("%(name)-12s: %(levelname)-8s %(message)s")
    console_stdout.setFormatter(formatter)
    console_stderr.setFormatter(formatter)

    # add the handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_stdout)
    root_logger.addHandler(console_stderr)


def reset_logging() -> None:
    """Reset loggers.

    This is only needed during testing, as otherwise the loggers are not
    destroyed resulting in the same output multiple time. This is taken
    from:

    https://stackoverflow.com/a/56810619
    """
    manager = logging.root.manager
    manager.disable = logging.NOTSET
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger) and LIBRARY_LOGGER_NAME in str(logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                # Copied from `logging.shutdown`.
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)


P = ParamSpec("P")


def log_init_args(method: Callable[P, None]) -> Callable[P, None]:
    """Log arguments of __init__ method.

    Args:
        method: __init__ method
    Returns:
        Decorated __init__ method
    """

    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        signature = inspect.signature(method)
        default_kwargs = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        all_keys = list(signature.parameters.keys())
        args_as_kwargs = {all_keys[i]: args[i] for i in range(len(args))}
        all_kwargs = dict(default_kwargs, **args_as_kwargs, **kwargs)

        def key_fun(pair: tuple[str, Any]) -> int:
            if pair[0] in all_keys:
                return all_keys.index(pair[0])
            return len(all_keys)

        all_kwargs = dict(sorted(all_kwargs.items(), key=key_fun))

        _logger = logging.getLogger(args[0].__module__)
        _logger.info(get_str_table(args[0].__class__.__name__, all_kwargs, use_repr=True))
        method(*args, **kwargs)

    return wrapper


def setup_logger_on_worker(
    name: str = "worker", log_dir: None | Path = None, level: int = logging.INFO
) -> logging.Logger:
    """Setup a logger on a scheduler's worker.

    Args:
        name: Name of the logger.
        log_dir: Path to the directory for the log file.
        level: Logging level.

    Returns:
        logger: Logger instance.
    """
    logger = logging.getLogger(name)

    formatter = NewLineFormatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M"
    )

    logger.setLevel(level)
    # if the worker is set up for the first time
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Remove all FileHandlers, keep others (like StreamHandler)
    # it should always have at least the StreamHandler from above but for safety
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()  # optional: closes the file descriptor

    if log_dir is not None:
        log_file = log_dir / "worker.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
