#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# BLE specification: 
# https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-62/out/en/low-energy-controller/radio-physical-layer-specification.html

def plot_spectrogram_waterfall(
            spectrogram_db,
            time_axis_ms,
            freq_axis_mhz,
            output_path,
            title           = "Spectrogram Waterfall",
            low_percentile  = 5,
            high_percentile = 99,
            dpi             = 150,
            line_frequencies_mhz = None):

    low         = np.percentile(spectrogram_db, low_percentile)
    high        = np.percentile(spectrogram_db, high_percentile)
    clipped     = np.clip(spectrogram_db, low, high)
    image_data  = clipped.T

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im = ax.matshow(
        image_data,
        aspect="auto",
        origin="lower",
        extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_mhz[0], freq_axis_mhz[-1]],
        cmap="viridis",
        vmin=low,
        vmax=high,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")
    if line_frequencies_mhz is not None:
        for line_freq_mhz in line_frequencies_mhz:
            ax.axhline(
                y=line_freq_mhz,
                color="red",
                linewidth=1.5,
                linestyle="-",
            )
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.savefig(output_path, dpi=dpi)
    plt.show()

def calculate_stft(input_data, sample_rate_hz, center_freq_hz, nfft=1024, overlap=256):
    if len(input_data) < nfft:
        raise ValueError("Input capture is too short for the selected FFT size.")

    window         = np.hanning(nfft).astype(np.float32)
    num_frames     = 1 + (len(input_data) - nfft) // overlap
    spectrogram_db = np.empty((num_frames, nfft), dtype=np.float32)

    for frame_idx in range(num_frames):
        start      = frame_idx * overlap
        frame      = input_data[start:start + nfft] * window
        spectrum   = np.fft.fftshift(np.fft.fft(frame, n=nfft))

        spectrogram_db[frame_idx] = 20.0 * np.log10(np.abs(spectrum) + 1e-12)

    freq_axis_mhz   = ( np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz)) + center_freq_hz) / 1e6
    time_axis_ms    = (np.arange(num_frames) * overlap / sample_rate_hz) * 1e3

    return spectrogram_db, time_axis_ms, freq_axis_mhz

def demod_fm(data, factor = 1):
    angle = np.unwrap(np.angle(data))
    #d_angle = angle[1:] - angle[:-1]
    d_angle = angle[:-1] - angle[1:]
    return d_angle * factor

def create_remez_lowpass_fir(
            input_sample_rate_hz,
            passband_hz,
            passband_ripple_db,
            stopband_hz,
            stopband_attenuation_db):

    if not (0.0 < passband_hz < stopband_hz < input_sample_rate_hz * 0.5):
        raise ValueError("Expected 0 < passband < stopband < Nyquist.")

    passband_ripple_linear = (10 ** (passband_ripple_db / 20.0) - 1.0) / (10 ** (passband_ripple_db / 20.0) + 1.0)
    stopband_ripple_linear = 10 ** (-stopband_attenuation_db / 20.0)

    transition_bw_hz        = stopband_hz - passband_hz
    num_taps_estimate       = int(np.ceil((stopband_attenuation_db - 8.0) / (2.285 * (2.0 * np.pi * transition_bw_hz / input_sample_rate_hz))))
    num_taps                = max(3, num_taps_estimate | 1)

    fir_taps = signal.remez(
        numtaps = num_taps,
        bands   = [0.0, passband_hz, stopband_hz, input_sample_rate_hz * 0.5],
        desired = [1.0, 0.0],
        weight  = [1.0 / passband_ripple_linear, 1.0 / stopband_ripple_linear],
        fs      = input_sample_rate_hz,
    )

    return fir_taps


# Input data contains 8-bit I/Q values, with I and Q interleaved.
# The data is sampled at 96 Msps with a center frequency of 2441 MHz.
with open("ble/advertising-sequence.cs8", "rb") as f:
    ble_input  = np.fromfile(f, dtype=np.int8)

# Reduce to [-1:1] range
ble_input  = ble_input.astype(np.complex64) / 128.0

# Merge alternating real/imaginary values into 1 complex value
ble_input  = ble_input[::2] #+ ble_input[1::2] * 1j 

print(len(ble_input))

# BLE occupies a spectrum from 2400 MHz to 2483.5 MHz.
# There are 40 channels that are each 2 MHz wide. 
# The center frequency is 2402 + k * 2 MHz.

# Build a spectrogram waterfall diagram.
sample_rate_hz  = 96e6
center_freq_hz  = 2441e6

if False:
    nfft            = 1024
    overlap         = 256
    
    ble_input_waterfall, time_axis_ms, freq_axis_mhz = calculate_stft(
        input_data      = ble_input,
        sample_rate_hz  = sample_rate_hz,
        center_freq_hz  = center_freq_hz,
        nfft            = nfft,
        overlap         = overlap,
    )
    
    print("Number of STFT frames:", len(ble_input_waterfall))
    
    output_path      = "ble_input_data_waterfall.png"
    channel_boundary_lines_mhz = np.arange(2401.0, freq_axis_mhz[-1] + 2.0, 2.0)
    plot_spectrogram_waterfall(
        spectrogram_db  = ble_input_waterfall,
        time_axis_ms    = time_axis_ms,
        freq_axis_mhz   = freq_axis_mhz,
        output_path     = output_path,
        title           = "BLE Advertising Sequence Waterfall",
        line_frequencies_mhz = channel_boundary_lines_mhz,
        )
    
    print(f"Saved spectrogram waterfall to {output_path}")
    print(
        f"Time axis: {time_axis_ms[0]:.6f} to {time_axis_ms[-1]:.6f} ms, "
        f"Frequency axis: {freq_axis_mhz[0]:.3f} to {freq_axis_mhz[-1]:.3f} MHz"
    )

# The data was sampled after heterodyning from 2441 MHz to 0 MHz.
# However, the channel center frequencies of BLE are at 2402, 2404, ... 2440, 2442, ... MHz
# and they are now at -39 MHz, -37 MHz, ... -1, +1, ... MHz.
# For a basic channelizer, we don't want this 1 MHz offset, so do a complex heterodyne of 1 MHz
# to make all center channels a multiple of 2, without offset.

n                       = np.arange(len(ble_input), dtype=np.float32)
heterodyne_1mhz         = np.exp(1j * 2.0 * np.pi * 1e6 * n / sample_rate_hz).astype(np.complex64)
ble_input_1mhz_shift    = ble_input * heterodyne_1mhz

h_lpf = create_remez_lowpass_fir(
    input_sample_rate_hz     = sample_rate_hz,
    passband_hz              = 600e3,
    passband_ripple_db       = 1.0,
    stopband_hz              = 800e3,
    stopband_attenuation_db  = 50.0
    )

print("h_lpf nr taps:", len(h_lpf))

# There are 40 channels that are 2 MHz wide, but instead of a 80 MHz sample rate, we have
# 96. So the decimation factor is 48 instead of 40.
decim_factor    = int(lp_input_sample_rate_hz // 2e6)

# Pad the filter with zeros so that the polyphase decomposition 
# is a clean 2D array.
#
# -1 % 10 = 9 so using the module of a negative number is a neat way
# to calculate the amount of padding.
h_lpf       = np.pad(h_lpf, (0, -len(h_lpf) % decim_factor) )
print("H_lpf padded nr taps:", len(h_lpf))

# Polyphase decomposition: 48 rows, each row has interleaved coefficients.
# Given the following input array [0, 1, 2, 3, 4, 4, 5, ...],
# first split it into a 2D array like this:
# [ [ 0, 1, 2, 3],
#   [ 4, 5, 6, 7], 
# ...
# then transpose so that it looks like this:
# [ [ 0, 4, 8, 12, ... ]
#   [ 1, 5, 9, 13, ... ]
# ...
h_lpf_poly  = np.reshape(h_lpf, ( (len(h_lpf) // decim_factor), decim_factor) ).T

# Now do the same polyphase decomposition/decimation of the input signal
ble_input_1mhz_shift = np.pad(ble_input_1mhz_shift, (0, -len(ble_input_1mhz_shift) % decim_factor) )
ble_decim   = np.flipud(
    np.reshape(
        ble_input_1mhz_shift,
        ((len(ble_input_1mhz_shift) // decim_factor), decim_factor),
    ).T
)

# Calculate the output of all polyphase filters
h_poly_out  = np.array([np.convolve(ble_decim[_], h_lpf_poly[_]) for _ in range(decim_factor)])

# Use the IFFT to calculate the output of all channels
# The command below is the vectorized version of this:
# for col_idx in range(num_columns):
#     channel_data[:, col_idx] = np.fft.ifft(h_poly_out[:, col_idx])
channel_data  = np.fft.ifft(h_poly_out, axis=0).astype(np.complex64)

channel_sample_rate_hz   = sample_rate_hz / decim_factor
channel_time_ms          = np.arange(channel_data.shape[1]) / channel_sample_rate_hz * 1e3

chan_41_time_mask        = (channel_time_ms >= 2.4) & (channel_time_ms <= 2.8)
chan_41_time_ms          = channel_time_ms[chan_41_time_mask]
chan_41                  = channel_data[41, chan_41_time_mask]
chan_41_fm               = demod_fm(chan_41, factor = 1)

chan_33_time_mask        = (channel_time_ms >= 1.13) & (channel_time_ms <= 1.23)
chan_33_time_ms          = channel_time_ms[chan_33_time_mask]
chan_33                  = channel_data[33, chan_33_time_mask]
chan_33_fm               = demod_fm(chan_33, factor = 1)

fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=True)
axs[0].plot(chan_33_time_ms, chan_33.real, label="I")
axs[0].plot(chan_33_time_ms, chan_33.imag, label="Q")
axs[0].plot(chan_33_time_ms, np.abs(chan_33), label="|IQ|", linewidth=1.2)
axs[0].set_title("Channel 33 Time Plot (2.4 ms to 2.8 ms)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True, alpha=0.3)
axs[0].legend()
axs[0].margins(x=0.0)
axs[1].plot(chan_33_time_ms[1:], chan_33_fm, color="tab:green", label="FM Decoded")
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("FM")
axs[1].grid(True, alpha=0.3)
axs[1].legend()
axs[1].margins(x=0.0)
fig.savefig("chan_33_time_plot.png", dpi=150)
plt.show()
