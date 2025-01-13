"""Base class for RL capabilities."""

import logging
import time

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from queens.iterators.iterator import Iterator
from queens.models.rl_model import RLModel
from queens.utils.logger_settings import log_init_args

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
    def __init__(self, model, parameters, global_settings, agent='PPO', policy='MlpPolicy', total_timesteps=10000, agent_kwargs=None):
        """Initialize an RLIterator.

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
        super().__init__(model, parameters, global_settings)

        # Check that a valid agent has been provided
        agent = agent.upper()
        agent_class = _supported_agents.get(agent, None)
        if agent_class is None:
            raise ValueError(f'Unsupported agent: {agent}')
        
        # Check that the provided policy is compatible with the chosen agent
        if policy not in agent_class.policy_aliases.keys():
            raise ValueError(
                f'Unsupported policy: `{policy}` for agent `{agent}`!\n'
                f'Agent {agent} only supports the following policies: '
                f'{list(agent_class.policy_aliases.keys())}.'
            )
        
        # Check that a valid model has been provided
        if not issubclass(model, RLModel):
            raise ValueError(
                'Unsupported model:\n'
                '`RLIterator` only supports models that inherit from `RLModel`.'
            )
        
        # instantiate the stable-baselines3 agent
        self.agent = agent_class(policy, model, **agent_kwargs)

        self.total_timesteps = total_timesteps


    def core_run(self):
        """Core run of RLIterator."""
        _logger.info("Welcome to Reinforcement Learning core run.")
        start = time.time()
        self.agent.learn(self.total_timesteps)
        end = time.time()
        _logger.info("Agent training took %E seconds.", end - start)