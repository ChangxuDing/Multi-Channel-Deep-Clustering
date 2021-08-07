# -*- coding: utf-8
from bss_model.utils import EPSILON
import numpy as np
import librosa

EPSILON=np.finfo(np.float32).eps


def sin_win(fftl):
    window = np.array([np.sin(np.pi * n / (fftl - 1)) for n in range(fftl)])
    return window

def stft_sample(sample,
         frame_length=512,
         frame_shift=64,
         center=True,
         window="hann",
         apply_log=False,
         transpose=True,
         pad_mode="constant"):
    stft_samples = []
    for i in range(4):
        stft_mat = librosa.stft(
            sample[i, :],
            frame_length,
            frame_shift,
            frame_length,
            window=window,
            center=center)
        if apply_log:
            stft_mat = np.log(np.abs(stft_mat))
        if transpose:
            stft_mat = np.transpose(stft_mat)
        stft_samples.append(stft_mat)
    
    return np.stack(stft_samples,axis=0), sample.shape[1]

def istft(stft_mat,
          frame_length=512,
          frame_shift=64,
          center=True,
          window="hann",
          transpose=True,
          norm=True,
          nsamps=None):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    samps = librosa.istft(
        stft_mat,
        frame_shift,
        frame_length,
        window=window,
        center=center,
        length=nsamps)
    # renorm if needed
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / (samps_norm+EPSILON)
    samps = samps/np.max(np.abs(samps))*0.7
    # same as MATLAB and kaldi
    # samps_int16 = (samps * MAX_INT16).astype(np.int16)
    return samps

