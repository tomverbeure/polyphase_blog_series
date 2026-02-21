#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

from signal_gen import SigInfo, generate_signal

SIGNAL1_FREQ_MHZ    = 22
SIGNAL2_FREQ_MHZ    = 17
SIGNAL3_FREQ_MHZ    = 29
SIGNAL4_FREQ_MHZ    = 30

SIGNAL1_AMPL_DB     = 0
SIGNAL2_AMPL_DB     = -10
SIGNAL3_AMPL_DB     = -20
SIGNAL4_AMPL_DB     = -30
FLOOR_NOISE_AMPL_DB = -50
OOB_NOISE_AMPL_DB   = -20

SAMPLE_CLOCK_MHZ    = 100
NR_SAMPLES          = 2048

LO_FREQ_MHZ         = 20

LABEL_OFFSET_MHZ    = 1

LPF_FIR_TAPS        = 201
LPF_PASSBAND_MHZ    = 5
LPF_KAISER_BETA     = 12.0

DECIM_FACTOR        = 10

# Stopband filter to create out-of-band noise
OOB_NOISE_FIR_TAPS              = 301
OOB_NOISE_STOPBAND_CENTER_MHZ   = 20
OOB_NOISE_STOPBAND_WIDTH_MHZ    = 14
OOB_NOISE_KAISER_BETA           = 14.0

sample_clock_hz     = SAMPLE_CLOCK_MHZ * 1e6
signal1_freq_hz     = SIGNAL1_FREQ_MHZ * 1e6
signal2_freq_hz     = SIGNAL2_FREQ_MHZ * 1e6
signal3_freq_hz     = SIGNAL3_FREQ_MHZ * 1e6
signal4_freq_hz     = SIGNAL4_FREQ_MHZ * 1e6
lo_freq_hz          = LO_FREQ_MHZ      * 1e6

#============================================================
# Generate test signal
#============================================================
signal_info = SigInfo(
    nr_samples         = NR_SAMPLES,
    sample_rate        = sample_clock_hz,
    frequencies_hz     = [signal1_freq_hz, signal2_freq_hz],
    amplitudes_db      = [SIGNAL1_AMPL_DB, SIGNAL2_AMPL_DB],
    stopband_center_hz = OOB_NOISE_STOPBAND_CENTER_MHZ * 1e6,
    stopband_width_hz  = OOB_NOISE_STOPBAND_WIDTH_MHZ * 1e6,
    noise_floor_db     = FLOOR_NOISE_AMPL_DB,
    oob_noise_db       = OOB_NOISE_AMPL_DB,
    oob_filter_taps    = OOB_NOISE_FIR_TAPS,
    oob_kaiser_beta    = OOB_NOISE_KAISER_BETA,
)

t, signal = generate_signal(signal_info)

signal_multi_info = SigInfo(
    nr_samples         = NR_SAMPLES,
    sample_rate        = sample_clock_hz,
    frequencies_hz     = [signal1_freq_hz, signal2_freq_hz, signal3_freq_hz, signal4_freq_hz],
    amplitudes_db      = [SIGNAL1_AMPL_DB, SIGNAL2_AMPL_DB, SIGNAL3_AMPL_DB, SIGNAL4_AMPL_DB],
    stopband_center_hz = 25e6,
    stopband_width_hz  = 20e6,
    noise_floor_db     = FLOOR_NOISE_AMPL_DB,
    oob_noise_db       = OOB_NOISE_AMPL_DB,
    oob_filter_taps    = OOB_NOISE_FIR_TAPS,
    oob_kaiser_beta    = OOB_NOISE_KAISER_BETA,
)

t, signal_multi = generate_signal(signal_multi_info)

#============================================================
# Perform all DSP operations
#============================================================

# Low-pass FIR filter: sinc window method (via firwin)
fir_cutoff          = LPF_PASSBAND_MHZ / (SAMPLE_CLOCK_MHZ / 2.0)
h_lpf               = firwin(LPF_FIR_TAPS, fir_cutoff, window=("kaiser", LPF_KAISER_BETA), pass_zero=True)

# Real BPF
tap_idx             = np.arange(LPF_FIR_TAPS)
h_bpf_real          = h_lpf * (2.0 * np.cos(2 * np.pi * lo_freq_hz * tap_idx / sample_clock_hz))

signal_bpf_real     = np.convolve(signal, h_bpf_real, mode="same")
signal_bpf_decim_real = signal_bpf_real[::DECIM_FACTOR]

# Complex BPF
complex_lo          = np.exp(1j * 2 * np.pi * lo_freq_hz * tap_idx / sample_clock_hz)
h_bpf_complex       = h_lpf * complex_lo

signal_bpf_complex  = np.convolve(signal, h_bpf_complex, mode="same")
signal_bpf_decim_complex = signal_bpf_complex[::DECIM_FACTOR]

sample_clock_decim_hz = sample_clock_hz / DECIM_FACTOR

#------------------------------------------------------------
# polyphase channelizer
#------------------------------------------------------------

# Convert h_lpf into its polyphase decomposition.
h_poly              = np.zeros((DECIM_FACTOR, int(np.ceil(LPF_FIR_TAPS / DECIM_FACTOR))))
for phase in range(DECIM_FACTOR):
    phase_taps      = h_lpf[phase::DECIM_FACTOR]
    h_poly[phase, :len(phase_taps)] = phase_taps

# Decimate the input signal into different signals, each with a different phase.
signal_multi_decim  = np.zeros((DECIM_FACTOR, int(np.ceil(NR_SAMPLES/DECIM_FACTOR))))
for phase in range(DECIM_FACTOR):
    phase_decim     = signal_multi[DECIM_FACTOR-1-phase::DECIM_FACTOR]
    signal_multi_decim[phase, :len(phase_decim)] = phase_decim

# Apply the polyphase sub-filters to their corresponding decimated input stream
h_poly_out          = np.zeros((DECIM_FACTOR, len(signal_multi_decim[0])))
for phase in range(DECIM_FACTOR):
    phase_h_out     = np.convolve(signal_multi_decim[phase], h_poly[phase], mode="same")
    h_poly_out[phase, :len(phase_h_out)] = phase_h_out

# For each output time step, select the 10 results from each phase and apply and 
# do an IFFT on them. 
signal_poly_out    = np.zeros((DECIM_FACTOR, int(np.ceil(NR_SAMPLES/DECIM_FACTOR))), dtype=complex)
for m in range(len(h_poly_out[0])):
    ifft_input  = h_poly_out[:, m]
    ifft_out    = np.fft.ifft(ifft_input)
    signal_poly_out[:, m] = ifft_out

#============================================================
# Plot: input spectrum + real BPF
#============================================================

# I didn't include the real version of the BPF in the blog...
if False:
    window = np.kaiser(NR_SAMPLES, 14.0)
    coherent_gain_real = np.sum(window) / 2.0
    fft_vals = np.fft.fftshift(np.fft.fft(signal * window))
    freqs_hz = np.fft.fftshift(np.fft.fftfreq(NR_SAMPLES, d=1.0 / sample_clock_hz))

    mag = np.abs(fft_vals) / coherent_gain_real
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    mag_db -= np.max(mag_db)

    bpf_fft_vals = np.fft.fftshift(np.fft.fft(h_bpf_real, n=NR_SAMPLES))
    bpf_mag = np.abs(bpf_fft_vals)
    bpf_mag_db = 20 * np.log10(np.maximum(bpf_mag, 1e-12))
    bpf_mag_db -= np.max(bpf_mag_db)

    fig_bpf, ax_bpf = plt.subplots(1, 1, figsize=(8, 4))
    ax_bpf.plot(freqs_hz / 1e6, mag_db)
    ax_bpf.plot(freqs_hz / 1e6, bpf_mag_db, color="tab:green")
    ax_bpf.set_xlabel("Frequency (MHz)")
    ax_bpf.set_ylabel("Magnitude (dB)")
    ax_bpf.set_title("Spectrum of input signal and bandpass filter")
    ax_bpf.grid(True)
    ax_bpf.set_xlim(freqs_hz[0] / 1e6, freqs_hz[-1] / 1e6)
    ax_bpf.set_ylim(-80, None)

    fig_bpf.tight_layout()
    fig_bpf.savefig("polyphase_het_sim-bpf.svg", format="svg")
    fig_bpf.savefig("polyphase_het_sim-bpf.png", format="png", dpi=200)
    plt.show()

    #============================================================
    # Plot: bandpass filtered signal spectrum
    #============================================================

    fig_bpf_time, ax_bpf_time = plt.subplots(1, 1, figsize=(8, 4))
    window_bpf = np.kaiser(NR_SAMPLES, 14.0)
    coherent_gain_bpf = np.sum(window_bpf) / 2.0
    fft_bpf_vals = np.fft.fftshift(np.fft.fft(signal_bpf_real * window_bpf))
    freqs_bpf_hz = np.fft.fftshift(np.fft.fftfreq(NR_SAMPLES, d=1.0 / sample_clock_hz))
    mag_bpf = np.abs(fft_bpf_vals) / coherent_gain_bpf
    mag_bpf_db = 20 * np.log10(np.maximum(mag_bpf, 1e-12))
    mag_bpf_db -= np.max(mag_bpf_db)

    ax_bpf_time.plot(freqs_bpf_hz / 1e6, mag_bpf_db)
    ax_bpf_time.set_xlabel("Frequency (MHz)")
    ax_bpf_time.set_ylabel("Magnitude (dB)")
    ax_bpf_time.set_title("Bandpass filtered signal")
    ax_bpf_time.grid(True)
    ax_bpf_time.set_xlim(freqs_bpf_hz[0] / 1e6, freqs_bpf_hz[-1] / 1e6)
    ax_bpf_time.set_ylim(-80, None)

    fig_bpf_time.tight_layout()
    fig_bpf_time.savefig("polyphase_het_sim-signal_bfp_filtered.svg", format="svg")
    fig_bpf_time.savefig("polyphase_het_sim-signal_bfp_filtered.png", format="png", dpi=200)
    plt.show()

    #============================================================
    # Plot: decimated bandpass filtered signal spectrum
    #============================================================

    window_bpf_decim = np.kaiser(len(signal_bpf_decim_real), 14.0)
    coherent_gain_bpf_decim = np.sum(window_bpf_decim) / 2.0
    fft_bpf_decim_vals = np.fft.fftshift(np.fft.fft(signal_bpf_decim_real * window_bpf_decim))
    freqs_bpf_decim_hz = np.fft.fftshift(
        np.fft.fftfreq(len(signal_bpf_decim_real), d=1.0 / sample_clock_decim_hz)
    )
    mag_bpf_decim = np.abs(fft_bpf_decim_vals) / coherent_gain_bpf_decim
    mag_bpf_decim_db = 20 * np.log10(np.maximum(mag_bpf_decim, 1e-12))
    mag_bpf_decim_db -= np.max(mag_bpf_decim_db)

    fig_bpf_decim, ax_bpf_decim = plt.subplots(1, 1, figsize=(8, 4))
    ax_bpf_decim.plot(freqs_bpf_decim_hz / 1e6, mag_bpf_decim_db)
    ax_bpf_decim.set_xlabel("Frequency (MHz)")
    ax_bpf_decim.set_ylabel("Magnitude (dB)")
    ax_bpf_decim.set_title("Decimated real bandpass filtered signal")
    ax_bpf_decim.grid(True)
    ax_bpf_decim.set_xlim(freqs_bpf_decim_hz[0] / 1e6, freqs_bpf_decim_hz[-1] / 1e6)
    ax_bpf_decim.set_ylim(-80, None)

    fig_bpf_decim.tight_layout()
    fig_bpf_decim.savefig("polyphase_het_sim-signal_bfp_filtered_decim_real.svg", format="svg")
    fig_bpf_decim.savefig("polyphase_het_sim-signal_bfp_filtered_decim_real.png", format="png", dpi=200)
    plt.show()

#============================================================
# Plot: input spectrum + complex BPF
#============================================================

if True:
    bpf_c_fft_vals = np.fft.fftshift(np.fft.fft(h_bpf_complex, n=NR_SAMPLES))
    bpf_c_mag = np.abs(bpf_c_fft_vals)
    bpf_c_mag_db = 20 * np.log10(np.maximum(bpf_c_mag, 1e-12))
    bpf_c_mag_db -= np.max(bpf_c_mag_db)
    
    window_bpf_c = np.kaiser(NR_SAMPLES, 14.0)
    coherent_gain_bpf_c = np.sum(window_bpf_c)
    fft_bpf_c_vals = np.fft.fftshift(np.fft.fft(signal_bpf_complex * window_bpf_c))
    freqs_bpf_c_hz = np.fft.fftshift(np.fft.fftfreq(NR_SAMPLES, d=1.0 / sample_clock_hz))
    mag_bpf_c = np.abs(fft_bpf_c_vals) / coherent_gain_bpf_c
    mag_bpf_c_db = 20 * np.log10(np.maximum(mag_bpf_c, 1e-12))
    mag_bpf_c_db -= np.max(mag_bpf_c_db)
    
    fig_bpf_c, axs_bpf_c = plt.subplots(2, 1, figsize=(8, 8))
    
    axs_bpf_c[0].plot(freqs_hz / 1e6, mag_db)
    axs_bpf_c[0].plot(freqs_hz / 1e6, bpf_c_mag_db, color="tab:green")
    axs_bpf_c[0].set_xlabel("Frequency (MHz)")
    axs_bpf_c[0].set_ylabel("Magnitude (dB)")
    axs_bpf_c[0].set_title("Spectrum of input signal and complex bandpass filter")
    axs_bpf_c[0].grid(True)
    axs_bpf_c[0].set_xlim(freqs_hz[0] / 1e6, freqs_hz[-1] / 1e6)
    axs_bpf_c[0].set_ylim(-80, None)
    
    axs_bpf_c[1].plot(freqs_bpf_c_hz / 1e6, mag_bpf_c_db)
    axs_bpf_c[1].set_xlabel("Frequency (MHz)")
    axs_bpf_c[1].set_ylabel("Magnitude (dB)")
    axs_bpf_c[1].set_title("Complex bandpass filtered signal")
    axs_bpf_c[1].grid(True)
    axs_bpf_c[1].set_xlim(freqs_bpf_c_hz[0] / 1e6, freqs_bpf_c_hz[-1] / 1e6)
    axs_bpf_c[1].set_ylim(-80, None)
    
    fig_bpf_c.tight_layout()
    fig_bpf_c.savefig("polyphase_het_sim-bpf_complex_filtered.svg", format="svg")
    fig_bpf_c.savefig("polyphase_het_sim-bpf_complex_filtered.png", format="png", dpi=200)
    plt.show()

#============================================================
# Plot: decimated complex bandpass filtered signal spectrum
#============================================================

# This code does straight decimation instead of first heterodyning
# back to 0... because the result is the same.
if True:
    window_bpf_decim_c = np.kaiser(len(signal_bpf_decim_complex), 14.0)
    coherent_gain_bpf_decim_c = np.sum(window_bpf_decim_c)
    fft_bpf_decim_c_vals = np.fft.fftshift(np.fft.fft(signal_bpf_decim_complex * window_bpf_decim_c))
    freqs_bpf_decim_c_hz = np.fft.fftshift(
        np.fft.fftfreq(len(signal_bpf_decim_complex), d=1.0 / sample_clock_decim_hz)
    )
    mag_bpf_decim_c = np.abs(fft_bpf_decim_c_vals) / coherent_gain_bpf_decim_c
    mag_bpf_decim_c_db = 20 * np.log10(np.maximum(mag_bpf_decim_c, 1e-12))
    mag_bpf_decim_c_db -= np.max(mag_bpf_decim_c_db)
    
    fig_bpf_decim_c, ax_bpf_decim_c = plt.subplots(1, 1, figsize=(8, 4))
    ax_bpf_decim_c.plot(freqs_bpf_decim_c_hz / 1e6, mag_bpf_decim_c_db)
    ax_bpf_decim_c.set_xlabel("Frequency (MHz)")
    ax_bpf_decim_c.set_ylabel("Magnitude (dB)")
    ax_bpf_decim_c.set_title("Decimated complex bandpass filtered signal")
    ax_bpf_decim_c.grid(True)
    ax_bpf_decim_c.set_xlim(freqs_bpf_decim_c_hz[0] / 1e6, freqs_bpf_decim_c_hz[-1] / 1e6)
    ax_bpf_decim_c.set_ylim(-80, None)
    
    fig_bpf_decim_c.tight_layout()
    fig_bpf_decim_c.savefig("polyphase_het_sim-signal_bfp_filtered_decim_complex.svg", format="svg")
    fig_bpf_decim_c.savefig("polyphase_het_sim-signal_bfp_filtered_decim_complex.png", format="png", dpi=200)
    plt.show()

#============================================================
# Plot: spectrum of multiple channel signal
#============================================================

if True:
    window_multi = np.kaiser(NR_SAMPLES, 14.0)
    coherent_gain_multi = np.sum(window_multi) / 2.0
    fft_multi_vals = np.fft.fftshift(np.fft.fft(signal_multi * window_multi))
    freqs_multi_hz = np.fft.fftshift(np.fft.fftfreq(NR_SAMPLES, d=1.0 / sample_clock_hz))
    mag_multi = np.abs(fft_multi_vals) / coherent_gain_multi
    mag_multi_db = 20 * np.log10(np.maximum(mag_multi, 1e-12))
    mag_multi_db -= np.max(mag_multi_db)
    
    fig_signal_multi, ax_signal_multi = plt.subplots(1, 1, figsize=(8, 4))
    ax_signal_multi.plot(freqs_multi_hz / 1e6, mag_multi_db)
    ax_signal_multi.set_xlabel("Frequency (MHz)")
    ax_signal_multi.set_ylabel("Magnitude (dB)")
    ax_signal_multi.set_title("Spectrum of Multiple Channel Signal")
    ax_signal_multi.grid(True)
    ax_signal_multi.set_xlim(freqs_multi_hz[0] / 1e6, freqs_multi_hz[-1] / 1e6)
    ax_signal_multi.set_ylim(-80, None)
    for freq_mhz in (-5, 5, 15, 25, 35, 45):
        ax_signal_multi.axvline(freq_mhz, color="tab:red", linestyle=":", linewidth=2.0)
    
    fig_signal_multi.tight_layout()
    fig_signal_multi.savefig("polyphase_het_sim-signal_multi_spectrum.svg", format="svg")
    fig_signal_multi.savefig("polyphase_het_sim-signal_multi_spectrum.png", format="png", dpi=200)
    plt.show()

#============================================================
# Plot: spectrum of polyphase output channels
#============================================================

if True:
    window_poly_ch = np.kaiser(len(signal_poly_out[2]), 14.0)
    coherent_gain_poly_ch = np.sum(window_poly_ch)
    freqs_poly_hz = np.fft.fftshift(
        np.fft.fftfreq(len(signal_poly_out[2]), d=1.0 / sample_clock_decim_hz)
    )
    
    fft_poly_ch1_vals = np.fft.fftshift(np.fft.fft(signal_poly_out[1] * window_poly_ch))
    mag_poly_ch1 = np.abs(fft_poly_ch1_vals) / coherent_gain_poly_ch
    
    fft_poly_ch2_vals = np.fft.fftshift(np.fft.fft(signal_poly_out[2] * window_poly_ch))
    mag_poly_ch2 = np.abs(fft_poly_ch2_vals) / coherent_gain_poly_ch
    
    fft_poly_ch3_vals = np.fft.fftshift(np.fft.fft(signal_poly_out[3] * window_poly_ch))
    mag_poly_ch3 = np.abs(fft_poly_ch3_vals) / coherent_gain_poly_ch
    
    mag_poly_ch1_db = 20 * np.log10(np.maximum(mag_poly_ch1, 1e-12))
    mag_poly_ch2_db = 20 * np.log10(np.maximum(mag_poly_ch2, 1e-12))
    mag_poly_ch3_db = 20 * np.log10(np.maximum(mag_poly_ch3, 1e-12))
    
    mag_poly_ref_db = max(
        np.max(mag_poly_ch1_db),
        np.max(mag_poly_ch2_db),
        np.max(mag_poly_ch3_db),
    )
    mag_poly_ch1_db -= mag_poly_ref_db
    mag_poly_ch2_db -= mag_poly_ref_db
    mag_poly_ch3_db -= mag_poly_ref_db
    
    fig_poly_out, (ax_poly_ch1, ax_poly_ch2, ax_poly_ch3) = plt.subplots(3, 1, figsize=(8, 9))
    
    ax_poly_ch1.plot(freqs_poly_hz / 1e6, mag_poly_ch1_db)
    ax_poly_ch1.set_xlabel("Frequency (MHz)")
    ax_poly_ch1.set_ylabel("Magnitude (dB)")
    ax_poly_ch1.set_title("Spectrum of Channel 1")
    ax_poly_ch1.grid(True)
    ax_poly_ch1.set_xlim(freqs_poly_hz[0] / 1e6, freqs_poly_hz[-1] / 1e6)
    ax_poly_ch1.set_ylim(-80, 5)
    
    ax_poly_ch2.plot(freqs_poly_hz / 1e6, mag_poly_ch2_db)
    ax_poly_ch2.set_xlabel("Frequency (MHz)")
    ax_poly_ch2.set_ylabel("Magnitude (dB)")
    ax_poly_ch2.set_title("Spectrum of Channel 2")
    ax_poly_ch2.grid(True)
    ax_poly_ch2.set_xlim(freqs_poly_hz[0] / 1e6, freqs_poly_hz[-1] / 1e6)
    ax_poly_ch2.set_ylim(-80, 5)
    
    ax_poly_ch3.plot(freqs_poly_hz / 1e6, mag_poly_ch3_db)
    ax_poly_ch3.set_xlabel("Frequency (MHz)")
    ax_poly_ch3.set_ylabel("Magnitude (dB)")
    ax_poly_ch3.set_title("Spectrum of Channel 3")
    ax_poly_ch3.grid(True)
    ax_poly_ch3.set_xlim(freqs_poly_hz[0] / 1e6, freqs_poly_hz[-1] / 1e6)
    ax_poly_ch3.set_ylim(-80, 5)
    
    fig_poly_out.tight_layout()
    fig_poly_out.savefig("polyphase_het_sim-signal_poly_out_ch2_ch3.svg", format="svg")
    fig_poly_out.savefig("polyphase_het_sim-signal_poly_out_ch2_ch3.png", format="png", dpi=200)
    plt.show()
