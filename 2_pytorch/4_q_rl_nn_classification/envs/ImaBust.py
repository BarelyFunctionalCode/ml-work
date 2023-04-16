import numpy as np

from .OneToOne import OneToOneEnv

class ImaBustEnv(OneToOneEnv):
  def __init__(self, obs_size_d1, min_value, max_value, program_interface):
    super().__init__(obs_size_d1, min_value, max_value)

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
    new_obs, data = self.program_interface.apply_action(action)
    new_obs_flattened = new_obs.flatten()

    # No action, just getting base state for resetting env
    if action is None:
      return new_obs, None, None

    # Setting initial terminal state
    terminated = False

    action_reward_small_value_good = (
      (
        (
          (
            1 -
            ( # Get ratio of current value by max possible value
              data["action_value"] /
              self.max_value
            ) # Flip ratio
          ) - 0.5 # Scaled from -0.5 - 0.5 instead of 0 - 1
        ) * 2 # Scaled from -1 - 1
      ) * (np.mean(new_obs_flattened) / (self.max_value - 1)) # Multiply by Average ratio
    ) + (1 - (np.mean(new_obs_flattened) / (self.max_value - 1))) # For smaller values, add a portion of the Average ratio


    # action_reward_small_value_good = (
    #   (
    #     1 -
    #     ( # Get ratio of current value by max possible value
    #       data["action_value"] /
    #       self.max_value
    #     ) # Flip ratio
    #   ) - 0.5 # Scaled from -0.5 - 0.5 instead of 0 - 1
    # ) * 2 # Scaled from -1 - 1

    min_steps_incentive_reward_scaled = (
      (
        (
          1 - 
          ( # Get ratio of current steps by max possible steps
            self.steps / # Current steps
            (
              (self.obs_size_d1 ** 2) * (self.max_value - self.min_value - 2) # Max possible steps
            )
          ) # Flip ratio
        ) ** (1. / 2) # Square Root it
      ) - 0.5 # Scaled from -0.5 - 0.5 instead of 0 - 1
    ) * 2 # Scaled from -1 - 1



    # more_steps_more_reward = (
    #   self.steps / # Current steps
    #   (
    #     (self.obs_size_d1 ** 2) * (self.max_value - self.min_value - 2) # Max possible steps
    #   )
    # )
    
    # Total reward
    reward = action_reward_small_value_good
    # reward = (action_reward_small_value_good * 0.6) + (min_steps_incentive_reward_scaled * 0.4)


    # Check if cell is at winning value
    if data["action_value"] == self.max_value - 1:
      reward = 2.0

    # Check win condition (if all cell values equal self.max_value)
    if np.array_equal(new_obs_flattened, np.full((self.obs_size_d1 ** 2), self.max_value - 1)):
      reward = 10.0
      terminated = True
      self.wins += 1

    # Check lose condition (selecting a cell already at max value)
    if data["action_value"] >= self.max_value:
      reward = -2.0
      terminated = True

    reward_array = np.full((self.obs_size_d1 ** 2), 0.0)
    reward_array[data["action_coord"][0] * self.obs_size_d1 + data["action_coord"][1]] = reward

    # print("Reward:" + str(reward) + " | Action: " + str(data["action_row"]) + ", " + str(data["action_col"]) + " | Value: " + str(data["action_value"]) + " | Steps: " + str(self.steps) + " | Wins: " + str(self.wins))
    # print("action_reward_small_value_good: " + str(action_reward_small_value_good) + " min_steps_incentive_reward_scaled:" + str(min_steps_incentive_reward_scaled) + " | Action: " + str(data["action_row"]) + ", " + str(data["action_col"]) + " | Value: " + str(data["action_value"]) + " | Steps: " + str(self.steps) + " | Wins: " + str(self.wins))

    return new_obs, reward, terminated

  # Stop training if desired amount of wins have been achieved
  def get_success_count(self):
    return self.wins