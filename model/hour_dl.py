from torch.utils.data import Dataset
import torchaudio

import torch
import torchaudio
from torchaudio import transforms
from dataset import AudioUtil
from pyedflib import highlevel
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import xml.etree.ElementTree as ET

from classifier import AudioClassifier


class HourDS(Dataset):
  def __init__(self, data_path, sample_rate=48000, time_step=1, sample_len=8):
    aud, _, _ = highlevel.read_edf(str(data_path), ch_names=['Mic'])
    self.aud = torch.from_numpy(aud[0, :]).float()
    self.sample_rate = sample_rate
    self.time_step = time_step
    self.sample_len = sample_len
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return int((self.aud.shape[0] / self.sample_rate - self.sample_len) / self.time_step)
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):

    start_second = idx * self.time_step
    stop_second = idx * self.time_step + self.sample_len
    sample = self.aud[
      self.sample_rate * start_second:self.sample_rate * stop_second
    ]
    sgram = AudioUtil.spectro_gram(
      (sample, self.sample_rate),
      n_mels=64, 
      n_fft=1024, 
      hop_len=None
    )
    return sgram.to(torch.device("cuda")), 0

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
myModel.load_state_dict(torch.load('model_weights.pth'))
myModel.eval()
myModel = myModel.to(torch.device("cuda"))

val_ds = HourDS("/home/vaaru/Downloads/apnea/APNEA_EDF/00001642-100507/00001642-100507[001].edf")

val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

res = np.zeros((len(val_ds),), np.uint8)
for i, data in enumerate(val_dl):
    inputs = data[0].reshape(1, 1, 64, 751)
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
    pred = myModel(inputs)

    # _, res[i] = torch.max(pred, 1)
    res[i] = pred[0][0] > 0.5
    # print(f"{i}: {res[i]}")

np.save("preds.npy", res)

from matplotlib import pyplot as plt

avged = 1 - np.convolve(res.astype(np.float32), np.ones(20)) / 20

rml_file = "/home/vaaru/Downloads/apnea/APNEA_RML/00001642-100507.rml"
tree = ET.parse(rml_file)
events = tree.getroot()[5][1]

vlines_1 = []
vlines_2 = []
for event in events:
    if "Apnea" in event.attrib['Type']:
        event_type, start, duration = (event.attrib['Type'], event.attrib['Start'], event.attrib['Duration'])
        start, duration = float(start), float(duration)
        if(start+duration < 3600):
            vlines_1.append(start)
            vlines_2.append(start+duration)
           
plt.plot(avged)
plt.vlines(vlines_1, 0, 1, linestyles='dashed')
plt.vlines(vlines_2, 0, 1, linestyles='dashed')

plt.show()
