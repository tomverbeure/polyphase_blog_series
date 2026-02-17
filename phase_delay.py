#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

N = 1024
fs = 40e6
f_sine = 2e6
f_sine_2 = 30e6

# Time buffer: 1024 samples at 40 MHz
t = np.arange(N) / fs

# Single-tone complex sine wave at 2 MHz
x = np.exp(1j * 2 * np.pi * f_sine * t)

# Single-tone complex sine wave at 18 MHz
x2 = np.exp(1j * 2 * np.pi * f_sine_2 * t)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Pane 1: x[0:50]
axs[0].plot(np.arange(0, 50), np.real(x[0:50]), "-o", markersize=4)
axs[0].set_ylabel("Amp")
axs[0].set_title("x[0:50]")
axs[0].set_xlim(0, 49)
axs[0].set_xlabel("Sample index")
axs[0].grid(True, alpha=0.3)

# Pane 2: x[1:51]
axs[1].plot(np.arange(1, 51), np.real(x[1:51]), "-o", markersize=4)
axs[1].set_ylabel("Amp")
axs[1].set_title("x[1:51]")
axs[1].set_xlim(1, 50)
axs[1].set_xlabel("Sample index")
axs[1].grid(True, alpha=0.3)

# Pane 3: x[0:50] + x[1:51]
x_sum = x[0:50] + x[1:51]
axs[2].plot(np.arange(0, 50), np.real(x_sum), "-o", markersize=4)
axs[2].set_ylabel("Amp")
axs[2].set_title("x[0:50] + x[1:51]")
axs[2].set_xlim(0, 49)
axs[2].set_xlabel("Sample index")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Pane 1: x2[0:50]
axs[0].plot(np.arange(0, 50), np.real(x2[0:50]), "-o", markersize=4)
axs[0].set_ylabel("Amp")
axs[0].set_title("x2[0:50]")
axs[0].set_xlim(0, 49)
axs[0].set_xlabel("Sample index")
axs[0].grid(True, alpha=0.3)

# Pane 2: x2[1:51]
axs[1].plot(np.arange(1, 51), np.real(x2[1:51]), "-o", markersize=4)
axs[1].set_ylabel("Amp")
axs[1].set_title("x2[1:51]")
axs[1].set_xlim(1, 50)
axs[1].set_xlabel("Sample index")
axs[1].grid(True, alpha=0.3)

# Pane 3: x2[0:50] + x2[1:51]
x2_sum = x2[0:50] + x2[1:51]
axs[2].plot(np.arange(0, 50), np.real(x2_sum), "-o", markersize=4)
axs[2].set_ylabel("Amp")
axs[2].set_title("x2[0:50] + x2[1:51]")
axs[2].set_xlim(0, 49)
axs[2].set_xlabel("Sample index")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
