import torch
import torch.nn as nn
import struct
import numpy as np

inputs     = 9
outputnode = 3

model = nn.Sequential(
    nn.Linear(inputs, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, outputnode)
)

model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

X_mean = np.load('X_mean.npy')
X_std  = np.load('X_std.npy')
T_mean = np.load('T_mean.npy')
T_std  = np.load('T_std.npy')

layers = [l for l in model if isinstance(l, nn.Linear)]

with open('model_weights.bin', 'wb') as f:
    for v in X_mean: f.write(struct.pack('f', float(v)))
    for v in X_std:  f.write(struct.pack('f', float(v)))
    for v in T_mean: f.write(struct.pack('f', float(v)))
    for v in T_std:  f.write(struct.pack('f', float(v)))
    for layer in layers:
        W = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        f.write(struct.pack('ii', W.shape[0], W.shape[1]))
        f.write(W.astype('float32').tobytes())
        f.write(b.astype('float32').tobytes())

print("Exported → model_weights.bin")