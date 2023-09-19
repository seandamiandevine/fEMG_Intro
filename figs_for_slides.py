import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter, iirnotch
from scipy.ndimage import gaussian_filter1d

def bandpass_filter(x,lowcut, highcut, fs, order=5):
	nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, x)


def notch_filter(x, notch_freq, Q, fs):
    nyquist = 0.5 * fs
    notch_freq_normalized = notch_freq / nyquist
    b, a = iirnotch(notch_freq_normalized, Q)
    filtered_data = lfilter(b, a, x)
    return filtered_data

np.random.seed(2023)

# ****************************************************************************
# *                       ## Generate fake EMG signal                        *
# ****************************************************************************

# Parameters
dur = 10   # Duration of the signal in seconds
fs  = 1000 # Sampling rate in Hz
n   = int(dur * fs)
t   = np.linspace(0, dur, n)

# Generate a baseline signal with random noise
baseline_signal = 0.2 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(20, 15, n)
signal = baseline_signal.copy()

# Create muscle bursts in equal intervals
burst_idx = []
for start_t in [2,4,6,8]:
	start = np.argmin(np.abs(t-start_t))
    end   = start + np.random.randint(50, 450) # range [50,405]Hz
    amplitude = np.random.uniform(10, 50)
    signal[start:end] += amplitude * np.sin(2 * np.pi * np.random.uniform(10, 60) * t[start:end])
    burst_idx.append((start,end))


# Plot the synthetic EMG signal
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal)
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
[ax.axvline(i, ls='--',c='red') for i in [2,4,6,8]]
ax.grid(True,axis='y')
plt.show()
fig.savefig("figs/raw.png",transparent=True)
plt.close()


# ****************************************************************************
# *                          ## Cleaning the signal                          *
# ****************************************************************************

# 1. Bandpass filter
signal_bp = bandpass_filter(signal, 20, 450, fs) 

## plot bandpassed signal
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal, label="Raw")
ax.plot(t, signal_bp, label="Bandpassed",alpha=.7)
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.grid(True,axis='y')
ax.legend()
ax.axhline(0,c='red')
# plt.show()
fig.savefig("figs/bp.png",transparent=True)
plt.close()

# 1.5. BONUS: Guassian 1D filter
signal_gf = gaussian_filter1d(signal_bp, sigma=5)
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal_bp, label="Bandpassed")
ax.plot(t, signal_gf, label="Guassian 1D")
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.grid(True,axis='y')
ax.axhline(0,c='red')
ax.legend(loc="lower left")
# plt.show()
fig.savefig("figs/gaussian_1d.png",transparent=True)
plt.close()


# 2. Rectify the signal
signal_r = np.abs(signal_bp)

## plot recritfied signal
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal, label="Raw")
ax.plot(t, signal_bp, label="Bandpassed",alpha=.7)
ax.plot(t, signal_r, label="Rectified",alpha=.7)
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.grid(True,axis='y')
ax.axhline(0,c='red')
ax.legend()
# plt.show()
fig.savefig("figs/rect.png",transparent=True)
plt.close()


# 3. Notch filter @ 50Hz
signal_n50 = notch_filter(signal_r, 50, 10, fs)

## plot notch filtered signal
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal, label="Raw")
ax.plot(t, signal_bp, label="Bandpassed",alpha=.7)
ax.plot(t, signal_r, label="Rectified",alpha=.7)
ax.plot(t, signal_n50, label="Notch @ 50Hz",alpha=.7)
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.grid(True,axis='y')
ax.axhline(0,c='red')
ax.legend()
# plt.show()
fig.savefig("figs/notch.png",transparent=True)
plt.close()

# ****************************************************************************
# *                         # Summarizing the signal                         *
# ****************************************************************************

means    = np.zeros(len(burst_idx))
baseline = 750 # how far before start do we take as baseline
for idx in range(len(burst_idx)):
	start,end = burst_idx[idx]
	means[idx] = signal_n50[start:end].mean() - signal_n50[(start-baseline):start].mean()


## plot notch filtered signal only
fig, ax = plt.subplots(1, figsize=(10,4))
ax.plot(t, signal_n50)
ax.set(title="", xlabel="Time (s)", ylabel="Amplitude (mV)")
ax.grid(True)
[ax.axvline(i, ls='--',c='red') for i in [2,4,6,8]]
ax.legend()
plt.show()
fig.savefig("figs/notch_only.png",transparent=True)
plt.close()

## plot summary values
fig, ax = plt.subplots(1, figsize=(4,4))

sns.barplot(x=["Low Reward","High Reward","High Reward","Low Reward"], y=means, color='black', ax=ax)
ax.set(title="Mean(Signal) - Mean(Baseline)")
plt.show()
fig.savefig("figs/summary.png",transparent=True)


# ****************************************************************************
# *                 ## Selecting code from triggers and file                 *
# ****************************************************************************
## This code won't actually execute, but it shows you how you read from a file and parse it

# SAMPLING_RATE = 1_000

# data         = pd.read_csv("emg_file.txt", sep="\t")
# n_lines      = data.shape[0]
# signal_chan  = 1
# trigger_chan = np.arange(2,10)

# baseline = []
# signal   = []
# for line in range(n_lines):
# 	binary = ''.join(data.iloc[trigger_chan])
# 	if binary=="00000050":
# 		baseline.append(data.iloc[(line-SAMPLING_RATE):line , signal_chan])
# 		signal.append(data.iloc[(line+SAMPLING_RATE):line , signal_chan])











