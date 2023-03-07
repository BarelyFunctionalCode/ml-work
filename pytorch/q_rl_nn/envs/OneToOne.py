import gymnasium as gym
from gymnasium import spaces

### OneToOneEnv Environment
# Base class used for environments where the input nuerons directly map to the output nuerons (1 to 1)
# 
class OneToOneEnv(gym.Env):
  def __init__(self, obs_size, min_value, max_value):
    self.obs_size = obs_size
    self.min_value = min_value
    self.max_value = max_value
    self.steps = 0

    # Observations are values of all the cells on the board
    self.observation_space = spaces.Box(self.min_value, self.max_value, shape=(self.obs_size,), dtype=int)

    # Action is the index of the selected cell
    self.action_space = spaces.Discrete(self.obs_size)
  
  # Restart the program upon env reset (on terminal state)
  def _restart_program(self):
    raise NotImplementedError()

  # Apply selected action to the program
  def _apply_action(self, _action=None):
    raise NotImplementedError()

  # Stop training if desired metric has been achieved
  def get_success_count(self):
    raise NotImplementedError()

  # Reset env upon terminal state has been reached
  def reset(self, seed=None):
    # Restart program
    self._restart_program()

    # Reset Step Count
    self.steps = 0

    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Get the initial observation set and info of the program
    obs, _, _ = self._apply_action()

    return obs

  # Step taken in agent training loop
  def step(self, action):
    # Apply action and get new observation set and other data
    obs, reward, terminated = self._apply_action(action)

    # Increment step count
    self.steps += 1
    return obs, reward, terminated, False