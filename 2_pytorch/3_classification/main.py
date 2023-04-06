import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# CNN model
class CNN(nn.Module):
  # ((n-f+2p)/s)+1

  # where n is the pixels of the image i.e. 32
  # f is the number of kernels, in our case it is 5*5 kernel which mean f = 5
  # p is the padding, p = 0
  # s is the stride, s = 0
  def __init__(self, state_size_1d, options_count):
    super().__init__()
    self.kernal_size = 3
    self.out_channels = 20
    self.hidden_layer_ratio = 5
    self.conv_output_size = ((state_size_1d - self.kernal_size + 1) ** 2) * self.out_channels
    self.hidden_layer_size = self.conv_output_size * self.hidden_layer_ratio
    self.output_size = state_size_1d ** 2

    self.stack = nn.Sequential(
      nn.Conv2d(options_count, self.out_channels, self.kernal_size),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(self.conv_output_size, self.hidden_layer_size),
      nn.ReLU(),
      nn.Linear(self.hidden_layer_size, self.output_size),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    logits = self.stack(x)
    return logits


if __name__ == '__main__':
  # Hyperparameters
  min_value = 1
  max_value = 8
  state_size_1d = 10

  learning_rate = 1e-3
  epochs = 100

  dataset_count = 10000
  training_dataset_ratio = 0.8
  training_batch_ratio = 0.05
  training_batch_print_frequency = 10
  testing_batch_ratio = 0.1
  testing_print_count = 5

  training_batch_size = int(dataset_count * training_dataset_ratio * training_batch_ratio)
  testing_batch_size = int(dataset_count * (1 - training_dataset_ratio) * testing_batch_ratio)


  # Initialize Model
  options_count = max_value - min_value + 1
  model = CNN(state_size_1d, options_count).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = nn.NLLLoss()


  # Initialize Dataset

  # Generate data
  X = torch.tensor([], dtype=torch.int64, device=device)
  y = torch.tensor([], dtype=torch.int64, device=device)
  for i in range(dataset_count):
    new_X = torch.randint(min_value, max_value, (state_size_1d, state_size_1d), device=device).unsqueeze(0)
    new_y = new_X.argmin().unsqueeze(0)

    X = torch.cat((X, new_X))
    y = torch.cat((y, new_y))

  # Convert to one hot and reshape
  X = F.one_hot(X, num_classes=options_count).float()
  B, W, H, C = X.shape
  X = X.view(B, C, W, H)

  # Split into training and testing
  X_train = X[0:round(len(X) * training_dataset_ratio)]
  X_test = X[round(len(X) * training_dataset_ratio):]
  y_train = y[0:round(len(y) * training_dataset_ratio)]
  y_test = y[round(len(y) * training_dataset_ratio):]

  # Convert to pytorch dataloader
  trainset = TensorDataset(X_train, y_train)
  testset = TensorDataset(X_test, y_test)
  trainloader = DataLoader(trainset, batch_size=training_batch_size, shuffle=True)
  testloader = DataLoader(testset, batch_size=testing_batch_size, shuffle=False)


  # Training Loop
  model.train()
  for i in range(epochs):
    total_loss = 0.0
    for X_batch, y_batch in trainloader:
      # Forwards
      y_pred = model(X_batch)
      loss = loss_fn(y_pred, y_batch)

      # Backwards
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_value_(model.parameters(), 10)
      optimizer.step()
      total_loss += loss.item()
    batch_loss = total_loss / len(X_batch)
    if i % training_batch_print_frequency == 0: print(f"Training Loss: {batch_loss}")
  
  final_training_loss = batch_loss


  # Testing Model
  model.eval()
  total_loss = 0.0
  for i, (X_batch, y_batch) in enumerate(testloader):
    y_pred = model(X_batch)
    loss = loss_fn(y_pred, y_batch)
    total_loss += loss.item()

    if i == 0:
      print("Sample Test Run:")
      for X_set, y_set, y_pred_set in zip(X_batch[:testing_print_count], y_batch[:testing_print_count], y_pred[:testing_print_count]):
        C, W, H = X_set.shape
        X_set = X_set.view(W, H, C)
        X_set = X_set.argmax(dim=2)
        print(f"{X_set=}")
        print(f"{y_set=}")
        print(f"{y_pred_set=}")
        action = y_pred_set.argmax()
        print(f"{action=}")
  print(f"Average Test Run Loss: {total_loss / len(testloader)}")
  print(f"Final Training Loss: {final_training_loss}")