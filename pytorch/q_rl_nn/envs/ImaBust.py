import numpy as np

from .OneToOne import OneToOneEnv

class ImaBustEnv(OneToOneEnv):
  def __init__(self, obs_size, min_value, max_value, program_interface):
    super().__init__(obs_size, min_value, max_value)

    # TODO: replace this with a window handle and utilities for keyboard/mouse inputs to the program
    self.program_interface = program_interface

    # Track win count
    self.wins = 0
  
  # Restart the program upon env reset (on terminal state)
  def _restart_program(self):
    self.program_interface.restart()

  # Apply selected action to the program
  def _apply_action(self, action=None):
    # Apply the desired action to the prgram and get new state
    new_obs = self.program_interface.apply_action(action)

    # No action, just getting base state for resetting env
    if action is None:
      return new_obs, None, None

    # Setting initial terminal state
    terminated = False

    # The more evenly you fill out the board, the higher the reward
    # distrubution_incentive = float(np.interp(
    #   np.mean(new_obs)**2,
    #   (1,
    #   self.max_value**2),
    #   (0,
    #   1)
    # ))

    # Assigning reward, selecting smaller valued cells gives larger reward
    # action_reward_small_value_good = float(np.interp(
    #   ((self.max_value - new_obs[action]) + self.min_value)**3, # Raw Reward
    #   (self.min_value**3,       # Minimum possible raw reward
    #   self.max_value**3),       # Maximum possible raw reward
    #   (0,                       # Minimum possible scaled reward
    #   1)                        # Maximum possible scaled reward
    # ))

    # More steps you take, better reward
    max_steps_incentive_reward = np.sqrt(self.steps)

    # Selecting smaller values gives better reward
    action_reward_small_value_good_non_scaled = (self.max_value - new_obs[action]) + self.min_value

    # Total reward
    reward = np.float32(max_steps_incentive_reward + action_reward_small_value_good_non_scaled)
    # reward = distrubution_incentive
    # reward = (action_reward * 0.7) + (distrubution_incentive * 0.3)

    # Check win condition (if all cell values equal self.max_value)
    if np.array_equal(new_obs, np.full(self.obs_size, self.max_value)):
      # reward = 1.1
      terminated = True
      self.wins += 1

    # Check lose condition (selecting a cell already at max value)
    if new_obs[action] > self.max_value:
      # reward = -0.1
      reward = 0
      terminated = True

    return new_obs, reward, terminated

  # Stop training if desired amount of wins have been achieved
  def get_success_count(self):
    return self.wins