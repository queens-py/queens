"""Model for running RL tasks."""

import gymnasium as gym

from queens.models.model import Model
from queens.utils.logger_settings import log_init_args


class RLModel(Model, gym.Env):
    """TODO
    """

    @log_init_args
    def __init__(self):
        super().__init__()