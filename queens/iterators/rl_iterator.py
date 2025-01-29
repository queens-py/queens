"""Base class for RL capabilities."""

import logging
import time

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from queens.iterators.iterator import Iterator
from queens.models.rl_models.rl_model import RLModel
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


_supported_agents = {
    'A2C': A2C,
    'DDPG': DDPG,
    'DQN': DQN,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}


class RLIterator(Iterator):
    """Iterator for RL problems.
    
    Based on the *stable-baselines3* package [1].

    References:
        [1]: https://stable-baselines3.readthedocs.io/en/master/

    Attributes:
        agent (stable_baselines3.BaseAlgorithm): The RL algorithm to be used:

                            - ``A2C``: Advantage Actor Critic
                            - ``DDPG``: Deep Deterministic Policy Gradient
                            - ``DQN``: Deep Q-Network
                            - ``PPO``: Proximal Policy Optimization 
                            - ``SAC``: Soft Actor Critic
                            - ``TD3``: Twin Delayed Deep Deterministic Policy Gradient

    Returns:
        RLIterator (obj): Instance of the RLIterator class.
    """

    @log_init_args
    def __init__(self, model, parameters, global_settings, result_description=None, mode='training', interaction_steps=1000, initial_observation=None):
        """Initialize an RLIterator.

        TODO:
        - introduce a "mode" argument, that allows for "train" and "evaluation"
        - different core run depending on the mode

        Args:
            agent (str): String that defines the RL algorithm to be used:

                                - ``'A2C'``: Advantage Actor Critic- ``A2C``: Advantage Actor Critic
                                - ``'DDPG'``: Deep Deterministic Policy Gradient
                                - ``'DQN'``: Deep Q-Network
                                - ``'PPO'``: Proximal Policy Optimization 
                                - ``'SAC'``: Soft Actor Critic
                                - ``'TD3'``: Twin Delayed Deep Deterministic Policy Gradient
            policy (str): Policy class to use (MlpPolicy, CnnPolicy, ...)
            total_timesteps (int): The total number of timesteps to train the RL agent.
            agent_kwargs (dict): Additional arguments to pass to the stable-baselines3 agent.
        """
        # Make sure that a valid model has been provided
        if not isinstance(model, RLModel):
            raise ValueError(
                'Unsupported model:\n'
                '`RLIterator` only supports models that inherit from `RLModel`.'
            )
        
        super().__init__(model, parameters, global_settings)

        self.result_description = result_description

        if not mode in ['training', 'evaluation']:
            raise ValueError(
                f'Unsupported mode: {mode}\n'
                'The mode must be either `training` or `evaluation`.'
            )
        self._mode = mode
        self._interaction_steps = interaction_steps
        self._initial_observation = initial_observation

    @property
    def mode(self):
        """str: Mode of the RLIterator."""
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        """Set the mode of the RLIterator."""
        if not mode in ['training', 'evaluation']:
            raise ValueError(
                f'Unsupported mode: {mode}\n'
                'The mode must be either `training` or `evaluation`.'
            )
        self._mode = mode

    @property
    def interaction_steps(self):
        """int: Number of interaction steps to be performed."""
        return self._interaction_steps
    
    @interaction_steps.setter
    def interaction_steps(self, eval_steps):
        """Set the number of interaction steps."""
        if eval_steps < 0:
            raise ValueError(
                f'Unsupported number of interaction steps: {eval_steps}\n'
                'The number of interaction steps must be a positive integer.'
            )
        self._interaction_steps = eval_steps

    @property
    def initial_observation(self):
        """np.ndarray: Initial observation of the environment."""
        return self._initial_observation
    
    @initial_observation.setter
    def initial_observation(self, initial_observation):
        """Set the initial observation of the environment."""
        self._initial_observation = initial_observation

    def pre_run(self):
        """Pre run portion of RLIterator."""
        _logger.info("Initialize RLIterator.")

    def core_run(self):
        """Core run of RLIterator."""
        _logger.info("Welcome to Reinforcement Learning core run.")
        if self._mode == 'training':
            _logger.info("Starting agent training.")
            start = time.time()
            # Start the model training
            self.model.train()
            end = time.time()
            _logger.info("Agent training took %E seconds.", end - start)
        else: # self._mode == 'evaluation'
            _logger.info("Starting interaction loop.")
            if self._initial_observation is None:
                _logger.debug(
                    "No initial observation provided.\n"
                    "Resetting environment to generate an initial observation."
                )
                obs = self.model.get_initial_observation()
            else: # initial observation has been provided by the user
                _logger.debug("Using provided initial observation.")
                obs = self._initial_observation
            start = time.time()
            # Perform as many interaction steps as set by the user
            for _ in range(self._interaction_steps):
                obs = self.model.interact(obs)
            end = time.time()
            _logger.info("Interaction loop took %E seconds.", end - start)

    def post_run(self):
        """Post run portion of RLIterator."""
        if self.result_description:
            if self.result_description["write_results"]:
                _logger.info("Storing the trained agent for further processing.")
                write_results(
                    # TODO: How to access the trained agent?
                    self.global_settings.result_file(".pickle"),
                )