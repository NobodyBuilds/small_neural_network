import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import time
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

#params
inputs=9
nodes=18
N=1000 # data sample count
outputnode=3
lr=0.0001 #learning rate
epochs=10000
batch_size=8192


epoch_times = []

#load data from txt files
#files names
files = {
    'density':     ('data\\density.txt',     None),
    'neardensity': ('data\\neardensity.txt', None),
    'ncount':      ('data\\neighborcount.txt',      None),
    'position':    ('data\\position.txt',    ','),
    'velocity':    ('data\\velocity.txt',    ','),
    'target':      ('data\\target.txt',      ','),
}
#scans for the maximun number of rows each files shares
print("Checking data files for row counts...")
row_counts = {}
for key, (path, delim) in files.items():
    with open(path) as f:
        row_counts[key] = sum(1 for _ in f)
    print(f"{key}: {row_counts[key]} rows")

N = min(row_counts.values())
print(f"\nUsing N = {N} for dataset rows (minimum across files).")




def load_cached(txt_path, npy_path, delimiter=None, max_rows=None):
    if os.path.exists(npy_path):
        print(f"  cache hit: {npy_path}")
        data = np.load(npy_path)
        return data[:max_rows] if max_rows else data
    
    print(f"  first load: {txt_path}...")
    sep = delimiter if delimiter else '\\s+'
    df = pd.read_csv(txt_path, sep=sep, header=None, nrows=max_rows)
    data = df.values
    np.save(npy_path, data)
    print(f"  cached to {npy_path}")
    return data

#loads data
print("Loading data from files for {} rows...".format(N))

density     = load_cached('data\\density.txt',       'data\\density.npy',      max_rows=N)
neardensity = load_cached('data\\neardensity.txt',   'data\\neardensity.npy',  max_rows=N)
ncount      = load_cached('data\\neighborcount.txt', 'data\\ncount.npy',       max_rows=N)
pos         = load_cached('data\\position.txt',      'data\\position.npy',     ',', max_rows=N)
vel         = load_cached('data\\velocity.txt',      'data\\velocity.npy',     ',', max_rows=N)
target      = load_cached('data\\target.txt',        'data\\target.npy',       ',', max_rows=N)

print("Data loaded successfully.")

X = np.column_stack([
    density, neardensity, ncount,
    pos[:, 0], pos[:, 1], pos[:, 2],
    vel[:, 0], vel[:, 1], vel[:, 2],
])#stacking input in matrix (n,9) format

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std#normalized data

T_mean = target.mean(axis=0)
T_std  = target.std(axis=0)
T_std[T_std == 0] = 1
target = (target - T_mean) / T_std

np.save('data\\X_mean.npy', X_mean)
np.save('data\\X_std.npy',  X_std)
np.save('data\\T_mean.npy', T_mean)
np.save('data\\T_std.npy',  T_std)
print("Norm stats saved.")

X_t = torch.tensor(X, dtype=torch.float32).to(device)
T_t = torch.tensor(target, dtype=torch.float32).to(device)

dataset = TensorDataset(X_t, T_t)
loader  = DataLoader(dataset,
batch_size=batch_size,
shuffle=True
)


model = nn.Sequential(
     nn.Linear(inputs, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, outputnode),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn   = nn.MSELoss()

checkpoint_path = 'model.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Resumed from checkpoint.")
else:
    print("No checkpoint found, starting fresh.")


for e in range(epochs):
    epoch_start = time.time()
    epoch_loss  = 0

    for Xb, Tb in loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, Tb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    avg_time  = sum(epoch_times) / len(epoch_times)
    remaining = time.strftime("%H:%M:%S", time.gmtime(avg_time * (epochs - e - 1)))
    avg_loss  = epoch_loss / len(loader)

    print(f"{e+1:4d}/{epochs}  loss: {avg_loss:.6f}  ETA: {remaining}")
    if (e + 1) % 100 == 0:
        torch.save(model.state_dict(), f'model_epoch{e+1}.pth')
        print(f"  checkpoint saved → model_epoch{e+1}.pth")



torch.save(model.state_dict(), 'model.pth')
print("Model saved.")

model.eval()
with torch.no_grad():
    pred_norm = model(X_t).cpu().numpy()

pred_real = pred_norm * T_std + T_mean
outx = pred_real[:, 0]
outy = pred_real[:, 1]
outz = pred_real[:, 2]
print(f"Sample predictions:\n x:{outx[:5]}\n y:{outy[:5]}\n z:{outz[:5]}")