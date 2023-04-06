import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import math

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display
plt.ion()

class RewardSimulation(object):

  def __init__(self, obs_size, min_value, max_value):
    self.obs_size = obs_size
    self.min_value = min_value
    self.max_value = max_value
    self.steps = 0
    self.wins = 0

    self.state = np.random.randint(self.min_value, self.min_value + 3, (self.obs_size))

  def apply_action(self, action):
    self.state[action] += 1
    self.steps += 1

    return self._generate_reward(action)

  def _generate_reward(self, action):
    new_obs = self.state

    terminated = False


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
      reward = 2.0
      terminated = True
      self.wins += 1

    # Check lose condition (selecting a cell already at max value)
    if new_obs[action] > self.max_value:
      reward = -2.0
      terminated = True

    return reward, terminated




state_size = 25
min_value = 1
max_value = 8

sim = RewardSimulation(state_size, min_value, max_value)

results = []
terminated = False
while not terminated:
  random_index = np.random.choice(np.arange(state_size)[sim.state < max_value])
  if sim.state[random_index] == 8: continue

  reward, terminated = sim.apply_action(random_index)
  results.append(reward)


plt.figure(1)
plt.clf()
plt.title('Result')
plt.xlabel('Env Steps')
plt.ylabel('Reward')
plt.plot(results, label="Reward at Given Step")
plt.pause(0.001)
plt.legend(loc='upper left')
if is_ipython:
  display.display(plt.gcf())
plt.ioff()
plt.show()