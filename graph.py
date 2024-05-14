import time
import datetime
import json

import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import loadmat

import matplotlib.pyplot as plt

result_path="C:/Users/JD/OneDrive/바탕 화면/urp/results/noise2self_checkerboard/2024_05_05_00_22"
data = np.load(result_path+ ".npz")

result_path2="C:/Users/JD/OneDrive/바탕 화면/urp/4090trained/SSRL_Simsiam/ssrl_ex1_200"
data2= np.load(result_path2+ ".npz")

plt.figure(figsize=(10, 6))
losses = data['losses']
train_rmses = data['train_rmses']
psnrs = data['train_psnrs']
time_elapsed = data['time_elapsed']

f_losses = data2['f_losses']
g_losses = data2['g_losses']
train_rmses2 = data2['f_train_rmses']
train_rmses3 = data2['g_train_rmses']
time_elapsed2= data2['time_elapsed']
f_psnrs = data2['f_psnrs']
g_psnrs = data2['g_psnrs']


# don't touch
epochs = range(2, len(losses) + 1)
plt.subplot(1, 1, 1)
##


# b,r -> rgb
plt.plot(epochs,  losses[1:], 'b-', label='Noise2Self')
plt.plot(epochs, g_losses[1:], 'r-', label='SSRL_Simsiam_g')

# try
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Train losses")

plt.legend()
plt.tight_layout()
plt.show()

