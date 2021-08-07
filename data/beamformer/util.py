# -*- coding: utf-8
from bss_model.utils import EPSILON
import numpy as np
import librosa

EPSILON=np.finfo(np.float32).eps


def sin_win(fftl):
    window = np.array([np.sin(np.pi * n / (fftl - 1)) for n in range(fftl)])
    return window

def wav2spec(wav_data, shift, fftl):
    """
    padding with zero to avoid first and last frame change

    :param wav_data: microphone signals with shape mic_num*length
    :param shift: half of fftl
    :param fftl: fft length
    :return: sepctrums with shape num_frames*num_mic*(fftl//2+1)
    """
    mic_num, len_sample = np.shape(wav_data)
    if (len_sample % 2) != 0:
        raise ValueError("length of nfft better be 2^n")

    wav_data = wav_data / np.max(np.abs(wav_data))
    # zero-padding at the very begining and the end
    number_of_frame = (len_sample - fftl) // shift + 1
    wav_data = np.pad(wav_data,((0,0),(0,shift*number_of_frame+fftl-len_sample)),"constant",constant_values=(0,0))
    #window = np.hanning(fftl)
    window = sin_win(fftl)
    st = 0
    ed = fftl
    spectrums = np.zeros((mic_num, number_of_frame, fftl // 2 + 1), dtype=np.complex64)
    for i in range(number_of_frame):
        multi_signal_spectrum = np.fft.rfft(wav_data[:, (st + i*shift):(ed + i*shift)] * window, axis=1)
        spectrums[:, i, :] = multi_signal_spectrum

    return spectrums,len_sample


def spec2wav(spectrogram, sampling_frequency, fftl, shift_len,length):
    """
    :param spectrogram: num_frames*(fftl//2+1)
    :param sampling_frequency: sf
    :param fftl: fftl
    :param shift_len: shift
    :return: beam wav
    """
    num_frame = np.shape(spectrogram)[0]
    result = np.zeros(sampling_frequency * 60, dtype=np.float32)
    st = 0
    ed = fftl
    window = sin_win(fftl)
    for ii in range(num_frame):
        cut_data2 = np.real(np.fft.irfft(spectrogram[ii, :]))        
        result[(st + ii*shift_len):ed + (ii*shift_len)] += cut_data2 * window
    #used for zero-padding
    sig = result[: length]
    #sig = result[: ((num_frame-1)*shift_len+fftl)]
    return sig/ np.max(np.abs(sig)) *0.7


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

def search_peaks(num_src, P):
    """
    :param num_src: num of sources >=1
    :param P: output power
    :return:
    """
    if num_src == 1:
        src_idx = np.argmax(P)

    else:
        peak_idx = []
        n = P.shape[0]
        for i in range(n):
            # straightforward peak finding
            if P[i] >= P[(i - 1) % n] and P[i] > P[(i + 1) % n]:
                if len(peak_idx) == 0 or peak_idx[-1] != i - 1:
                    if not (i == n and P[i] == P[0]):
                        peak_idx.append(i)

        peaks = P[peak_idx]
        max_idx = np.argsort(peaks)[-num_src:]
        src_idx = [peak_idx[k] for k in max_idx]

    return src_idx



if __name__ == '__main__':

    sig = np.ones((6,4096))
    spec = wav2spec(sig,256,512)
    spec1= np.ones((17,257),dtype = np.complex)
    wav = spec2wav(spec1,16000,512,256 )