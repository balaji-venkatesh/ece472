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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=3)

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

      correct_prediction_val += ((prediction != 2) == (labels != 2)).sum().item()
      total_prediction_val += prediction.shape[0]

    num_batches_val = len(val_dl)
    avg_loss_val = running_loss_val / num_batches_val
    acc_val = correct_prediction_val/total_prediction_val

    print(f'Val Loss: {avg_loss_val:.2f}, Val Accuracy: {acc_val:.2f}')


# Create the model and put it on the GPU if available
myModel = AudioClassifier().bfloat16().load_state_dict(torch.load('model_weights.pth'))
device = torch.device("cuda")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

val_path = "/run/media/vaaru/T7/apnea/new_apnea_dataset/val"

val_ds = SoundDS(val_path)

val_dl = DataLoader(val_ds, batch_size=8192, shuffle=False)

evaluating(myModel, val_dl, num_epochs)

