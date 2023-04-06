import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random
import numpy as np


# Custom dataset for numpy arrays of floats
class FloatArrayDataset(Dataset):
  def __init__(self, data, truths, transform=None, target_transform=None):
    self.data = data
    self.truths = truths
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.truths)

  def __getitem__(self, idx):
    return self.data[idx], self.truths[idx]

  def generate_dataloader(self, batch_size):
    return DataLoader(self, batch_size=batch_size)

# Dataset:
#   X | {(batch_size * batch_count) input_count}: sample of random floats from min_output to max_output
#   y | {(batch_size * batch_count) output_count} : mean average of X multiplied by a fixed random float multiplied by sequential array of the length of outputs
def generate_random_float_array_dataset(batch_size, batch_count, input_count, output_count, min_output, max_output):
  print("Generating Dataset:")
  # Create random dataset based on desired parameters
  seed = random.random()
  data_sets = np.random.rand(batch_size * batch_count, input_count).astype(np.float32) * (max_output - min_output) + min_output
  data_sets_truths = (np.mean(data_sets, axis=1, keepdims=True) * np.full((len(data_sets), output_count), np.arange(1, output_count+1) * seed)).astype(np.float32)
  print(f"X Shape: {data_sets.shape}")
  print(f"y Shape: {data_sets_truths.shape}")

  # Put most of the dataset into training data
  training_data_sets = data_sets[:batch_size*(batch_count-1)]
  training_data_sets_truths = data_sets_truths[:batch_size*(batch_count-1)]
  train_dataloader = FloatArrayDataset(training_data_sets, training_data_sets_truths).generate_dataloader(batch_size)

  # A little leftover data for testing data
  testing_data_sets = data_sets[batch_size*(batch_count-1):]
  testing_data_sets_truths = data_sets_truths[batch_size*(batch_count-1):]
  test_dataloader = FloatArrayDataset(testing_data_sets, testing_data_sets_truths).generate_dataloader(batch_size)

  return train_dataloader, test_dataloader

# Used to train the model
def train(device, dataloader, model, loss_fn, optimizer, scheduler, print_results):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), (batch + 1) * len(X)
    if print_results: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  scheduler.step()

# Used to evaluate the model
def test(device, dataloader, model, loss_fn, print_results):
  size = len(dataloader.dataset) * len(dataloader.dataset[0])
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      # Compute evaluation error
      pred = model(X)
      test_loss += loss_fn(pred, y).item()

      # Compute accuracy
      # 1 - (error / max_error)
      correct += (
          torch.full((len(y), len(y[0])), 1).to(device)
            .sub(
              (pred - y) 
              .abs()
              .div(torch.max(torch.cat((pred, y), 0)) - torch.min(torch.cat((pred, y), 0)))
            )
        ).sum().item()
  
  # Normalize loss and accuracy
  test_loss /= num_batches
  correct /= size

  if print_results: print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  return pred, y