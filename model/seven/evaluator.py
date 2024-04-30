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
        conv1 = nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d(2)
        bn1 = nn.BatchNorm2d(100)
        init.kaiming_normal_(conv1.weight, a=0.1)
        conv1.bias.data.zero_()
        conv_layers += [conv1, relu1, pool1, bn1]

        for i in range(6):
          conv = nn.Conv2d(100, 100, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          relu = nn.ReLU()
          bn = nn.BatchNorm2d(100)
          init.kaiming_normal_(conv.weight, a=0.1)
          conv.bias.data.zero_()
          conv_layers += [conv, relu, bn]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=100, out_features=2)

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

        # Final output
        return x



# ----------------------------
# Training Loop
# ----------------------------
def evaluating(model, val_dl):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()

  running_loss_val = 0.0
  correct_prediction_val = 0.0
  total_prediction_val = 0.0

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
      # correct_prediction_val += ((prediction != 2) == (labels != 2)).sum().item()
      total_prediction_val += prediction.shape[0]

    num_batches_val = len(val_dl)
    avg_loss_val = running_loss_val / num_batches_val
    acc_val = correct_prediction_val/total_prediction_val

    print(f'Val Loss: {avg_loss_val:.2f}, Val Accuracy: {acc_val:.2f}')


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
myModel.load_state_dict(torch.load('model_weights_epoch9.pth'))
device = torch.device("cuda")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

val_path = "/run/media/vaaru/T7/apnea/new_apnea_dataset/val"

val_ds = SoundDS(val_path)

val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)

evaluating(myModel, val_dl)
