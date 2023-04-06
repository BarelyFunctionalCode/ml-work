import torch
import torch.nn.functional as F

import random
import time
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt

from model import initialize_model

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args):
    """Save a transition"""
    # if len(self.memory) % 1000 == 0: print(f"Replay Memory Size: {len(self.memory)}")
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


# Reinforment Learning Neural Network Agent
# Uses Replay Memory sampled in random batches and Fixed Q-Targets
class Agent(object):
  def __init__(self,
                b_load_saved_agent,
                state_size_1d,
                min_value,
                max_value,
                options_count,
                batch_size,
                loss_func,
                optimizer_option,
                learning_rate,
                gamma,
                tau,
                eps_decay,
                estimate_max_duration,
                success_threshold,
                device):

    self.b_load_saved_agent = b_load_saved_agent
    self.state_size_1d = state_size_1d
    self.min_value = min_value
    self.max_value = max_value
    self.options_count = options_count
    self.batch_size = batch_size
    self.loss_func = loss_func
    self.optimizer_option = optimizer_option
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.tau = tau
    self.eps_decay = eps_decay
    self.estimate_max_duration = estimate_max_duration
    self.success_threshold = success_threshold
    self.device = device

    # initialize models
    self.policy_net, self.optimizer = initialize_model(
                                        self.b_load_saved_agent,
                                        self.state_size_1d,
                                        self.options_count,
                                        self.optimizer_option,
                                        self.learning_rate,
                                        self.device
                                      )
    self.target_net, _ = initialize_model(
                            False,
                            self.state_size_1d,
                            self.options_count,
                            self.optimizer_option,
                            self.learning_rate,
                            self.device
                          )
    
    # Copy policy_net parameters to target_net
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # Initialize Replay Memory size
    self.memory = ReplayMemory(15000)

    # Initializing variables to contain plot graph metrics
    self.episode_env_success = []
    self.episode_eps = []
    self.episode_loss = []
    self.episode_durations = []

    # Initial eps threshold
    self.eps_threshold = 0.0

  # Choose either a random action, or learned action, depending on the current eps threshold
  def _select_action(self, state, env):
    # sample = random.random()
    
    # self.eps_threshold = max(self.eps_threshold * self.eps_decay, 0.05)
    # if sample > self.eps_threshold:
    self.policy_net.eval()
    with torch.no_grad():
      action_values = self.policy_net(state)
    self.policy_net.train()
    # print(f"learned action_values: {action_values}")
    return action_values
    # else:
    #   action_values = torch.tensor(np.array([env.action_space.sample()]), device=self.device, dtype=torch.float32)
    #   # action_values_softmax = F.log_softmax(action_values, dim=1)
    #   # state = torch.tensor(env.observation_space.sample(), dtype=torch.int64, device=self.device).unsqueeze(0)
    #   # state = F.one_hot(state, num_classes=self.options_count).float()
    #   # B, W, H, C = state.shape
    #   # state = state.view(B, C, W, H)
    #   # action_values = self.policy_net(state).detach()
    #   return action_values

  # Plot metrics relevent to the agent and env onto a graph for analysis
  def _plot_data(self, show_result=False):
    plt.figure(1)

    # Gather metrics
    eps_t = torch.tensor(self.episode_eps, dtype=torch.float)
    loss_t = torch.tensor(self.episode_loss, dtype=torch.float) / 4.0
    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    env_success_t = torch.tensor(self.episode_env_success, dtype=torch.float)

    plt.clf()
    if show_result:
      plt.title('Result')
    else:
      plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('EPS Threshold, Loss, Duration, Env Successes')

    # Plot metrics
    plt.plot(eps_t.numpy(), label="eps threshold (random or learned action taken)")
    plt.plot(loss_t.numpy(), label="loss (how bad it is)")
    plt.plot(durations_t.numpy(), label="durations (how long each episode lasted)")
    plt.plot(env_success_t.numpy(), label="env success (how close the agent is to being done)")

    # Take 100 episode averages and plot them too
    if len(loss_t) >= 100:
      l_means = loss_t.unfold(0, 100, 1).mean(1).view(-1)
      l_means = torch.cat((torch.zeros(99), l_means))
      plt.plot(l_means.numpy(), label="loss avg")
    if len(durations_t) >= 100:
      d_means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
      d_means = torch.cat((torch.zeros(99), d_means))
      plt.plot(d_means.numpy(), label="duration avg")


    time.sleep(0.001)  # pause a bit so that plots are updated
    plt.legend(loc='upper left')
    if is_ipython:
      if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
      else:
        display.display(plt.gcf())

  # After enough transitions are collected in the Replay Memory, optimize the policy
  def _optimize_model(self, env):
    if len(self.memory) < self.batch_size:
      return
    transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                     batch.next_state)), device=self.device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                     if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)

    self.policy_net.train()
    self.target_net.eval()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    action_values = self.policy_net(state_batch)
    # action_values = self.policy_net(state_batch, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # next_action_values = torch.zeros((len(action_batch), len(action_batch[0])), device=self.device)

    with torch.no_grad():
      expected_action_values = (self.policy_net(state_batch) + reward_batch).softmax(dim=1)
      # next_action_values = self.target_net(next_state_batch).detach()
      # next_action_values[non_final_mask, :] = self.target_net(non_final_next_states).detach()

    # Compute the expected Q values
    # expected_action_values = next_action_values.argmax(1)
    # expected_action_values = ((next_action_values * self.gamma) + reward_batch).softmax(dim=1)
    # expected_action_values = ((next_action_values * self.gamma) + reward_batch).argmax(1)

    # Compute loss
    loss_func_instance = self.loss_func()
    loss = loss_func_instance(action_values, expected_action_values)

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
    self.optimizer.step()

    return loss.item()

  # main training loop function
  def train(self, env):
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.int64, device=self.device).unsqueeze(0)
    state = F.one_hot(state, num_classes=self.options_count).float()
    B, W, H, C = state.shape
    state = state.view(B, C, W, H)

    # Loop training until env terminal state is reached, then optimize policy model and soft update target model
    for t in count():
      action = self._select_action(state, env)
      observation, reward, terminated, truncated = env.step(action.argmax(1).item())
      reward = torch.tensor(reward, device=self.device).unsqueeze(0)
      done = terminated or truncated

      # if terminated:
      #   next_state = None
      # else:
      next_state = torch.tensor(observation, dtype=torch.int64, device=self.device).unsqueeze(0)
      next_state = F.one_hot(next_state, num_classes=self.options_count).float()
      B, W, H, C = next_state.shape
      next_state = next_state.view(B, C, W, H)

      # Store the transition in memory
      # determin state importance, prioritize very high/low reward
          # abs([-0.1 - 1.1] - 0.5) * 2 = [0.0 - 1.2]
          # random.uniform(-0.5, 1) = [-0.5 - 1.0]
          # Gets all of the terminal states, and most of the middle transitions (about 60%)
      # if (abs(reward - 0.5) * 2) >= random.uniform(-0.5, 1):
      self.memory.push(state, action, next_state, reward)

      # Move to the next state
      state = next_state

      # Perform one step of the optimization (on the policy network)
      loss = self._optimize_model(env)

      # Soft update of the target network's weights
      # θ′ ← τ θ + (1 −τ )θ′
      target_net_state_dict = self.target_net.state_dict()
      policy_net_state_dict = self.policy_net.state_dict()
      for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
      self.target_net.load_state_dict(target_net_state_dict)

      # Terminal state reached, add metrics to plotting variables and restart training
      if done:
        self.episode_env_success.append(env.get_success_count() / self.success_threshold)
        self.episode_eps.append(min(self.eps_threshold, 1))
        self.episode_loss.append(loss if loss != None else 0)
        self.episode_durations.append((t + 1) / self.estimate_max_duration)
        self._plot_data()
        break

    return env.get_success_count()

  # Plot final results of agent and save agent's policy model data
  def finish_and_save(self):
    torch.save(self.policy_net.state_dict(), "model.pth")
    print("\n\nSaved PyTorch Model State to model.pth")
    self._plot_data(show_result=True)
    plt.ioff()
    plt.show()