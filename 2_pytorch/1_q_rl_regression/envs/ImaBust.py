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

    # Selecting smaller values gives better reward
    # action_reward_small_value_good_non_scaled = np.float32((self.max_value - new_obs[action]) + self.min_value)

    # More steps you take, better reward
    # max_steps_incentive_reward = np.sqrt(self.steps)

    # Assigning reward, selecting smaller valued cells gives larger reward
    # action_reward_small_value_good = np.float32(np.interp(
    #   ((self.max_value - new_obs[action]) + self.min_value)**3, # Raw Reward
    #   (self.min_value**3,       # Minimum possible raw reward
    #   self.max_value**3),       # Maximum possible raw reward
    #   (-1,                       # Minimum possible scaled reward
    #   1)                        # Maximum possible scaled reward
    # ))

    # Assigning reward, selecting smaller valued cells gives larger reward
    action_reward_small_value_good = (
                                      (
                                        (
                                          (
                                            1 -
                                            ( # Get ratio of current value by max possible value
                                              min((new_obs[action] - self.min_value) /
                                              (self.max_value - self.min_value), 1)
                                            ) # Flip ratio
                                          ) * 2 # Double it, and give it to the next person
                                        ) - 1 # Scaled from -1 - 1 instead of 0 - 2
                                      ) * (np.mean(new_obs) / self.max_value) # Multiply by Average ratio
                                    ) + (1 - (np.mean(new_obs) / self.max_value)) # For smaller values, add a portion of the Average ratio


    # More steps you take, worse reward
    min_steps_incentive_reward_scaled = (
                                          (
                                            (
                                              1 -
                                              ( # Get ratio of current steps by max possible steps
                                                min(self.steps /
                                                (self.obs_size * (self.max_value - 2)), 1)
                                              ) # Flip ratio
                                            ) ** (1. / 2) # Square Root it
                                          ) * 2 # Double it, and give it to the next person
                                        ) - 1 # Scaled from -1 - 1 instead of 0 - 2
    
    # Total reward
    # reward = action_reward_small_value_good
    # reward = min_steps_incentive_reward_scaled
    reward = (action_reward_small_value_good * 0.2) + (min_steps_incentive_reward_scaled * 0.8)

    # Always give good reward for turning a cell to the self.max_value
    # if new_obs[action] == self.max_value:
    #   reward += 0.3

    # Check win condition (if all cell values equal self.max_value)
    if np.array_equal(new_obs, np.full(self.obs_size, self.max_value)):
      reward = 100.0
      terminated = True
      self.wins += 1

    # Check lose condition (selecting a cell already at max value)
    if new_obs[action] > self.max_value:
      reward = -100.0
      terminated = True

    # print(f"{self.steps=} {new_obs[action]=} {reward=} {action_reward_small_value_good=} {min_steps_incentive_reward_scaled=}")
    return new_obs, reward, terminated

  # Stop training if desired amount of wins have been achieved
  def get_success_count(self):
    return self.wins