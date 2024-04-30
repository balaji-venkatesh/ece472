import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from dataset import SoundDS
# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.pool1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.pool2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=2)

        # Softmax
        self.sm = nn.Softmax(dim=1)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        x = self.sm(x)
        # Final output
        return x

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, val_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  weights = torch.Tensor([1.0, 1.5]).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    running_loss_val = 0.0
    correct_prediction_val = 0
    total_prediction_val = 0

    print(f"Training {epoch}")
    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Keep stats for Loss and Accuracy
      running_loss += loss.item()

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]

      if i % 50 == 49:    # print every 10 mini-batches
          print(f"\tBatch {i}, Train Loss: {running_loss / total_prediction:.2f}, Train Accuracy: {correct_prediction / total_prediction:.2f}")
    print("Done training {epoch}")
    with torch.no_grad():
      for i, data in enumerate(val_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss_val += loss.item()
        _, prediction = torch.max(outputs,1)

        correct_prediction_val += (prediction == labels).sum().item()
        total_prediction_val += prediction.shape[0]

    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction

    num_batches_val = len(val_dl)
    avg_loss_val = running_loss_val / num_batches_val
    acc_val = correct_prediction_val/total_prediction_val

    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.2f}, Train Accuracy: {acc:.2f}, Val Loss: {avg_loss_val:.2f}, Val Accuracy: {acc_val:.2f}')
    torch.save(myModel.state_dict(), f'model_weights_epoch{epoch}.pth')
  print('Finished Training')

if __name__ == "__main__":
  num_epochs = 100
  # Create the model and put it on the GPU if available
  myModel = AudioClassifier()
  device = torch.device("cuda")
  myModel = myModel.to(device)
  # Check that it is on Cuda
  next(myModel.parameters()).device

  train_path = "/run/media/vaaru/T7/apnea/new_apnea_dataset/train"
  val_path = "/run/media/vaaru/T7/apnea/new_apnea_dataset/val"

  train_ds = SoundDS(train_path)
  val_ds = SoundDS(val_path)

  train_dl = DataLoader(train_ds, num_workers=8, batch_size=24, shuffle=True)
  val_dl = DataLoader(val_ds, num_workers=1, batch_size=32, shuffle=True)

  training(myModel, train_dl, val_dl, num_epochs)
  torch.save(myModel.state_dict(), 'model_weights.pth')

