"""Model for training an RL agent in predefined gym environments."""

import logging

import gymnasium as gym

from queens.models.rl_models.rl_model import RLModel
from queens.models.rl_models.utils import supported_gym_environments
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class PredefinedGymModel(RLModel):
    """TODO
    """

    @log_init_args
    def __init__(self, env_name='CartPole-v1', env_kwargs={}, agent='PPO', policy='MlpPolicy', total_timesteps=10000, agent_kwargs={}, render_on_evaluation=False, render_args=()):
        """TODO
        """
        if env_name not in supported_gym_environments:
            raise ValueError(f'Environment `{env_name}` is not known to gymnasium')
        
        # If the environment name is known, create an environment instance
        self.env = gym.make(env_name, **env_kwargs)

        # Pass all arguments to the constructor of the parent class
        super().__init__(self.env, agent=agent, agent_options=agent_kwargs, policy=policy, total_timesteps=total_timesteps, render_on_evaluation=render_on_evaluation, render_args=render_args)

    def setup(self):
        """Setup the model for training."""
        _logger.info('Setting up the model for training')
        _logger.debug('Nothing to be setup for model `PredefinedGymModel`.')

    def _render(self, *render_args):
        """Render the current state of the environment."""
        self._vec_env.render(*render_args)