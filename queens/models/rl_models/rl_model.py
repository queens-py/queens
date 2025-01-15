"""Model for running RL tasks."""

import abc
import logging

from queens.models.model import Model
from queens.models.rl_models.utils import get_supported_sb3_policies, supported_sb3_agents
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class RLModel(Model):
    """TODO
    """

    @log_init_args
    def __init__(self, env, agent='PPO', policy='MlpPolicy', total_timesteps=10000, agent_kwargs={}, render_on_evaluation=False, render_args=()):
        """TODO"""
        super().__init__()

        # Check that a valid agent has been provided
        agent = agent.upper()
        agent_class = supported_sb3_agents.get(agent, None)
        if agent_class is None:
            raise ValueError(f'Unsupported agent: {agent}')
        
        # Check that the provided policy is compatible with the chosen agent
        supported_sb3_policies = get_supported_sb3_policies(agent_class)
        if policy not in supported_sb3_policies:
            raise ValueError(
                f'Unsupported policy: `{policy}` for agent `{agent}`:\n'
                f'Agent {agent} only supports the following policies: '
                f'{supported_sb3_policies}!'
            )

        # Store the environment instance (this is a gymnasium environment)
        self._env = env
        # Instantiate the stable-baselines3 agent
        self._agent = agent_class(policy, self._env, **agent_kwargs)
        # Retrieve a (vectorized) stable-baseline3 environment for evaluation
        self._vec_env = self._agent.get_env()

        self._total_timesteps = total_timesteps
        self._render_on_evaluation = render_on_evaluation
        self._render_args = render_args

    def evaluate(self, observation):
        """Interaction step of a (trained) RL agent with an environment.

        Args:
            observation (np.ndarray): The observation of the current state of 
                                      the environment.

        Returns:
            next_observation (dict): Dictionary with the model output 
                                     (corresponding to the observation of the 
                                     new state of the environment)
        """
        _logger.info(
            'Computing one agent-enviroment interaction (i.e., one timestep).'
        )
        action, _state = self._predict(observation)
        obs, reward, done, info = self._step(action)
        if self._render_on_evaluation:
            self._render(self._render_args)
        _logger.info('Interaction completed.')
        return obs # TODO wrap in dict
    
    def get_initial_observation(self):
        """TODO"""
        return self._vec_env.reset()
    
    def grad(self, samples, upstream_gradient):
        """TODO Check if this is always the case."""
        raise NotImplementedError(
            "Gradient information not available. \n"
            "If you need gradients, please use a different model or implement " 
            "the `grad` method in the child class."
        )
    
    def _predict(self, observation):
        """Make a single prediction with the trained RL agent.
        
        Args:
            observation (np.ndarray): Observation

        Returns:
            action (np.ndarray): Action to be executed based on the current 
                                 state of the environment
            state (np.ndarray): TODO
        """
        _logger.debug(
            'Predicting the next action based on the current state of the environment.'
        )
        return self._agent.predict(observation)
    
    @abc.abstractmethod
    def _render(self, *render_args):
        """Render the current state of the environment."""
        raise NotImplementedError(
            "I can't render anything for the currently selected model.\n"
            "If you want to render something, please use a different model "
            "or implement the `_render` method in the child class."
        )

    def _step(self, action):
        """Execute a single step in the environment.

        Args:
            action (np.ndarray): Action to be executed
        
        Returns:
            observation (np.ndarray): Observation of the new state of the environment
            reward (float): Reward obtained from the environment
            done (bool): Flag indicating whether the episode has finished
            info (dict): Additional information
        """
        _logger.debug('Applying an action to the environment.')
        return self._vec_env.step(action)
    
    def setup(self):
        """Setup the model."""
        raise NotImplementedError(
            "Nothing to setup for the currently selected model.\n"
            "If you need a setup step, please use a different model "
            "or implement the `setup` method in the child class."
        )
    
    def train(self):
        """Train the RL agent."""
        _logger.info(
            f'Training the RL agent for a total of {self._total_timesteps} timesteps.'
        )
        # Train the agent for the desired number of timesteps
        self._agent.learn(total_timesteps=self._total_timesteps)
        _logger.info('Training completed.')
