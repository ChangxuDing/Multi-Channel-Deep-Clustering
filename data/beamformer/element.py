import numpy as np
import librosa

class Element(object):

    def __init__(self,
                 num_mic=4,
                 sampling_frequency=16000,
                 fft_length=512,
                 fft_shift=256,
                 sound_speed=343,
                 theta_step=1,
                 frame_num=1000000):
        self.num_mic = num_mic
        self.mic_angle_vector = np.array([45, 315, 225, 135])
        # self.mic_angle_vector = np.array([315, 45, 225, 135])
        self.mic_diameter = 0.064
        self.sampling_frequency = sampling_frequency
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.sound_speed = sound_speed
        self.theta_step = theta_step
        self.frame_num = frame_num

    def get_sterring_vector(self, look_direction):
        '''
            return: sv of shape (N//2+1,num_mic)
        '''
        frequency_vector = librosa.fft_frequencies(self.sampling_frequency, self.fft_length)
        steering_vector = np.exp(1j * 2 * np.pi / self.sound_speed * self.mic_diameter / 2 *
                                 np.einsum("i,j->ij",frequency_vector, np.cos(np.deg2rad(look_direction) - np.deg2rad(
                                                 self.mic_angle_vector))))

        return steering_vector.T/self.num_mic

    def get_correlation_matrix(self, multi_signal):

        length = multi_signal.shape[1]
        frequency_grid = librosa.fft_frequencies(self.sampling_frequency, self.fft_length)
        R_mean = np.zeros((len(frequency_grid), self.num_mic, self.num_mic), dtype=np.complex64)

        num_frames = (length-self.fft_shift)//self.fft_shift

        if num_frames >= self.frame_num:
            num_frames = self.frame_num

        start = 0
        end = self.fft_length
        for _ in range(0, num_frames):
            multi_signal_cut = multi_signal[:, start:start + self.fft_length]
            complex_signal = np.fft.rfft(multi_signal_cut, axis=1)
            for f in range(0, len(frequency_grid)):
                R_mean[f, ...] += np.einsum("i,j->ij",complex_signal[:, f], np.conj(complex_signal[:, f]).T)
            start = start + self.fft_shift
            end = end + self.fft_shift

        return R_mean/num_frames

    def get_correlation_matrix_fb(self, multi_signal):

        length = multi_signal.shape[1]
        frequency_grid = librosa.fft_frequencies(self.sampling_frequency, self.fft_length)
        R_mean = np.zeros((len(frequency_grid), self.num_mic, self.num_mic), dtype=np.complex64)

        num_frames = self.frame_num

        start = 0
        end = self.fft_length
        for _ in range(0, num_frames):
            multi_signal_cut = multi_signal[:, start:start + self.fft_length]
            complex_signal = np.fft.fft(multi_signal_cut, axis=1)
            for f in range(len(frequency_grid)):
                R_mean[f, ...] += np.outer(complex_signal[:, f], np.conj(complex_signal[:, f]).T)
            start = start + self.fft_shift
            end = end + self.fft_shift

        start1 = length
        for _ in range(0, num_frames):
            multi_signal_cut1 = multi_signal[:, start1 - self.fft_length:start1]
            complex_signal = np.fft.fft(multi_signal_cut1, axis=1)
            for f in range(len(frequency_grid)):
                R_mean[f, ...] += np.outer(complex_signal[:, f], np.conj(complex_signal[:, f]).T)
            start1 = start1 - self.fft_shift

        return R_mean/num_frames/2
