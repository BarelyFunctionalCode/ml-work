import math
import numpy as np

from nn import MLP, Value

# network settings
step_limit = 400
learning_rate_start = 3.0
hidden_layer_size = 25
hidden_layer_count = 2

# training data
input_count = 10
output_count = 1
set_count = 50
training_data_sets = np.random.rand(set_count, input_count).astype(float) * 2 - 1
training_data_sets_truths = np.random.rand(set_count, output_count).astype(float) * 2 - 1

# initialize network
network = MLP(
  len(training_data_sets[0]), # number of input neurons
  np.concatenate([np.full(hidden_layer_count, hidden_layer_size), [output_count]]).tolist() # size/count of hidden/output layers
)

# training loop
for i in range(step_limit):
  # forward pass
  predictions_sets = list(map(network, training_data_sets))

  # "mean absolute error" loss
  data_loss = sum(
    abs(prediction - dataset_thruth)
    for dataset_thruths, predictions in zip(training_data_sets_truths, predictions_sets)
    for dataset_thruth, prediction in zip(dataset_thruths, predictions)
  ) / len(training_data_sets_truths.flatten())

  # L2 regularization
  alpha = 1e-4
  reg_loss = alpha * sum((p*p for p in network.parameters()))
  total_loss = data_loss + reg_loss

  # backward pass
  network.zero_grad()
  total_loss.backward()
  
  # update
  learning_rate = learning_rate_start - (learning_rate_start*0.9)*i/step_limit
  for parameter in network.parameters():
    parameter.data -= learning_rate * parameter.grad
  
  print(f"step {i} loss {total_loss.data}")

for dataset_thruths, predictions in zip(training_data_sets_truths, predictions_sets):
  print(f"\n\nTruth Value(s): {dataset_thruths} Prediction(s): {[prediction.data for prediction in predictions]}")