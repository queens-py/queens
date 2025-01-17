import inspect

import gymnasium as gym
import stable_baselines3 as sb3


def _create_supported_agents_dict():
    # Get all agent classes in the stable_baselines3 module
    supported_agents = {
        name: obj for name, obj in inspect.getmembers(sb3)
        if inspect.isclass(obj) and issubclass(obj, sb3.common.base_class.BaseAlgorithm)
    }
    return supported_agents

supported_sb3_agents = _create_supported_agents_dict()

def get_supported_sb3_policies(agent_class):
    if issubclass(agent_class, sb3.common.base_class.BaseAlgorithm):
        return list(agent_class.policy_aliases.keys())
    else:
        raise ValueError(
            f'{agent_class.__name__} is not a stable-baselines3 agent.\n'
            'Cannot find supported policies for this class!'
        )

supported_gym_environments = list(gym.envs.registry.keys())