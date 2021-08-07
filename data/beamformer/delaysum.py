# -*- coding: utf-8 -*-
import numpy as np
from beamformer import util
from beamformer.element import Element
import time
import os
import soundfile as sf

class Delaysum(Element):
    """
        calaulate the wav output with given direction and micphone inputs
    """
    
    def __init__(self, sr, frame_length, frame_shift):
        super().__init__()
        self.sampling_frequency = sr
        self.fft_length = frame_length
        self.fft_shift = frame_shift


    def apply_beamformer(self, steering_vector, complex_spectrum, length):
        """
        :param steering_vector: shape   num_mic*(fftl//2+1)
        :param complex_spectrum: shape  num_frames*num_mic*(fftl//2+1)
        :return:
        """
        num_mic, num_frames, fbins = np.shape(complex_spectrum)
        spec = np.zeros((num_frames, fbins), dtype=np.complex64)
        for f in range(0, fbins):
            spec[:, f] = np.dot(np.conjugate(steering_vector[:,f].T), complex_spectrum[:, :, f])
        return util.istft(spec,frame_length=self.fft_length, frame_shift=self.fft_shift,nsamps=length)

def run_ds(micro_data, direction, sr):
    """
    input:
        chunk: chunk read from 6 micros with size(chunk_size*8,)
        direction: predicted DOA
    return:
        beam signal
    """
    start = time.perf_counter()
    if sr==16000:
        das = Delaysum(16000, 512, 256)
    else:
        das = Delaysum(8000, 256, 128)
    sv = das.get_sterring_vector(direction)
    chunk_fft, length = util.stft_sample(micro_data, 
                                         frame_length=das.fft_length, 
                                         frame_shift=das.fft_shift)
    # print(chunk_fft.shape)
    wav = das.apply_beamformer(sv, chunk_fft, length)
    # print("ds耗时为{}".format(time.perf_counter()-start))
    return wav


if __name__ == '__main__':
    def multi_channel_read(prefix=r'./0-270/output_{}.wav',
                           channel_index_vector=np.array([0, 1, 2, 3, 4, 5])):
        wav, sr = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
        wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
        wav_multi[:, 0] = wav
        for i in range(1, len(channel_index_vector)):
            wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
        return wav_multi.T, sr
    start = time.perf_counter()
    multi_channels_data, sr = multi_channel_read()
    recon = run_ds(multi_channels_data,0)
    sf.write('../data/ds_0.wav', recon, 16000)
    print(time.perf_counter()-start)


