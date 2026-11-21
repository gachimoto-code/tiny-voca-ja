"""Mel spectrogram extraction for the vocoder pilot (pure PyTorch, no librosa).

This module DEFINES the mel spec for the whole mel+HifiGAN route: the
acoustic model in Step 2 must predict exactly this representation.

Spec: 24,000 Hz, n_fft 1024, hop 256, win 1024 (Hann), 80 mel bands,
fmin 0 Hz, fmax 12,000 Hz, Slaney-style mel scale and area-normalized
triangular filters, log compression log(max(1e-5, mel)).
"""
import torch

SR = 24000
N_FFT = 1024
HOP = 256
WIN = 1024
NUM_MELS = 80
FMIN = 0.0
FMAX = 12000.0

_mel_fb_cache = {}


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # Slaney-style: linear below 1 kHz, logarithmic above.
    f_sp = 200.0 / 3.0
    mels = hz / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = torch.log(torch.tensor(6.4)) / 27.0
    mask = hz >= min_log_hz
    mels[mask] = min_log_mel + torch.log(hz[mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    hz = mels * f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = torch.log(torch.tensor(6.4)) / 27.0
    mask = mels >= min_log_mel
    hz[mask] = min_log_hz * torch.exp(logstep * (mels[mask] - min_log_mel))
    return hz


def mel_filterbank(device="cpu", dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key in _mel_fb_cache:
        return _mel_fb_cache[key]
    fftfreqs = torch.linspace(0, SR / 2, N_FFT // 2 + 1)
    min_mel = _hz_to_mel(torch.tensor(FMIN))
    max_mel = _hz_to_mel(torch.tensor(FMAX))
    mels = torch.linspace(min_mel, max_mel, NUM_MELS + 2)
    hz_points = _mel_to_hz(mels)
    bin_points = hz_points  # in Hz, compare against fftfreqs directly
    fb = torch.zeros(NUM_MELS, N_FFT // 2 + 1)
    for i in range(NUM_MELS):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        up = (fftfreqs - left) / (center - left)
        down = (right - fftfreqs) / (right - center)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
        # Slaney area normalization
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        fb[i] *= enorm
    fb = fb.to(device=device, dtype=dtype)
    _mel_fb_cache[key] = fb
    return fb


_window_cache = {}


def mel_spectrogram(wav: torch.Tensor) -> torch.Tensor:
    """wav: [B, T] float in [-1, 1] -> log-mel [B, NUM_MELS, frames]."""
    if wav.dim() == 1:
        wav = wav[None, :]
    device = wav.device
    if device not in _window_cache:
        _window_cache[device] = torch.hann_window(WIN, device=device)
    window = _window_cache[device]
    pad = (N_FFT - HOP) // 2
    wav_p = torch.nn.functional.pad(wav, (pad, pad), mode="reflect")
    spec = torch.stft(
        wav_p, N_FFT, hop_length=HOP, win_length=WIN, window=window,
        center=False, pad_mode="reflect", normalized=False,
        onesided=True, return_complex=True,
    )
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    fb = mel_filterbank(device=device, dtype=wav.dtype)
    mel = torch.matmul(fb, mag)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel
