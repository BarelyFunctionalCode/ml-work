import torch

import random
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
                state_size,
                hidden_layer_count,
                hidden_layer_size,
                batch_size,
                loss_fn_option,
                optimizer_option,
                learning_rate,
                gamma,
                tau,
                eps_decay,
                estimate_max_duration,
                success_threshold,
                device):

    self.b_load_saved_agent = b_load_saved_agent
    self.state_size = state_size
    self.hidden_layer_count = hidden_layer_count
    self.hidden_layer_size = hidden_layer_size
    self.batch_size = batch_size
    self.loss_fn_option = loss_fn_option
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
                                        self.state_size,
                                        self.hidden_layer_count,
                                        self.hidden_layer_size,
                                        self.state_size,
                                        self.optimizer_option,
                                        self.learning_rate,
                                        self.device
                                      )
    self.target_net, _ = initialize_model(
                            False,
                            self.state_size,
                            self.hidden_layer_count,
                            self.hidden_layer_size,
                            self.state_size,
                            self.optimizer_option,
                            self.learning_rate,
                            self.device
                          )
    
    # Copy policy_net parameters to target_net
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # Initialize Replay Memory size
    self.memory = ReplayMemory(5000)

    # Initializing variables to contain plot graph metrics
    self.episode_env_success = []
    self.episode_eps = []
    self.episode_loss = []
    self.episode_durations = []

    # Initial eps threshold
    self.eps_threshold = 1.0

  # Choose either a random action, or learned action, depending on the current eps threshold
  def _select_action(self, state, env):
    sample = random.random()
    
    self.eps_threshold = max(self.eps_threshold * self.eps_decay, 0.01)
    if sample > self.eps_threshold:
      self.policy_net.eval()
      with torch.no_grad():
        action_values = self.policy_net(state).max(1)[1].view(1, 1)
      self.policy_net.train()
      return action_values
    else:
      return torch.tensor(np.array([[env.action_space.sample()]]), device=self.device, dtype=torch.long)

  # Plot metrics relevent to the agent and env onto a graph for analysis
  def _plot_data(self, show_result=False):
    plt.figure(1)

    # Gather metrics
    eps_t = torch.tensor(self.episode_eps, dtype=torch.float)
    loss_t = torch.tensor(self.episode_loss, dtype=torch.float)
    durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    env_success_t = torch.tensor(self.episode_env_success, dtype=torch.float)

    if show_result:
      plt.title('Result')
    else:
      plt.clf()
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


    plt.pause(0.001)  # pause a bit so that plots are updated
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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    self.policy_net.train()
    self.target_net.eval()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_action_values = torch.zeros(self.batch_size, device=self.device)

    with torch.no_grad():
      next_action_values[non_final_mask] = self.target_net(non_final_next_states).detach().max(1)[0]

    # Compute the expected Q values
    expected_action_values = (next_action_values * self.gamma) + reward_batch

    # Compute loss
    loss_fn = self.loss_fn_option()
    loss = loss_fn(action_values, expected_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    return loss.item()

  # main training loop function
  def train(self, env):
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    # Loop training until env terminal state is reached, then optimize policy model and soft update target model
    for t in count():
      action = self._select_action(state, env)
      observation, reward, terminated, truncated = env.step(action.item())
      reward = torch.tensor([reward], device=self.device)
      done = terminated or truncated

      if terminated:
        next_state = None
      else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

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
        self.episode_eps.append(self.eps_threshold)
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