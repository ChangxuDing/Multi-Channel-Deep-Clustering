# coding=utf-8

import numpy as np
import pickle
import os
import random
from scipy import signal
import soundfile as sf
import librosa
import rir_generator as rir
import multiprocessing as mp
from beamformer import delaysum as ds
import argparse

c = 343
fs = 16000
num_samples = 4096
sig_len = 4


def create_mix_pickle(spk_path, noise_path, output_path):
    mic_diameter = 0.064
    items = {"train": 2, "val": 2, "test": 2,}
    paths = ["train-clean-360", "dev-clean", "test-clean"]

    noise = ["tr", "cv", "tt"]
    for i, item in enumerate(items.keys()):
        config = []
        spk_path_item = spk_path + "/" + paths[i] + "/LibriSpeech/" + paths[i]
        output_path_item = output_path + "/" + item + "_16khz.pkl"
        noise_path_item = noise_path + "/" + noise[i]

        with open(output_path_item, "wb") as f:
            for uttr in range(items[item]):
                if uttr%10000==0:
                    print("finish {}".format(uttr))
                uttr_dic = {}
                """
                each uttr_dic contains：
                        speech：list contains spk1 and spk2 path
                        noise: noise path
                        overlap_ratio；An overlap ratio between the two speakers is uniformly sampled between 0% and 100% such
                                    that the average overlap ratio across the dataset is 50%
                        start_index: default 0
                        spk_snr： 0-5db
                        mic_pos_center； length(r,l-r) width(r,w-r)  height(1.5)
                        spk_d: distance between source and mic, (0.5, 1.5)
                        spk_angle:list, delta angle >30
                        spk_pos:spk(x,y,z)
                        noise_pos: random in room size
                        room_size:length and width (3,10), height(2.5, 4)
                        rt60:(0.2,0.7)
                        num_mic:4, for respeaker 4 mic
                """
                # spks path
                uttr_dic["spk"] = []
                path_dir = os.listdir(spk_path_item)
                samples = random.sample(path_dir, 2)
                for j, sample in enumerate(samples):
                    path_sample = spk_path_item + "/" + sample
                    path_sample_item = os.listdir(path_sample)

                    sample_audios_item = random.sample(path_sample_item, 1)
                    sample_audios_path = path_sample + "/" + sample_audios_item[0]
                    sample_audios_path_list = os.listdir(sample_audios_path)
                    for sample_audios_path_list_item in sample_audios_path_list:
                        if sample_audios_path_list_item.endswith("txt"):
                            sample_audios_path_list.remove(sample_audios_path_list_item)
                    sample_audio_item = sample_audios_path + "/" + random.sample(sample_audios_path_list[1:], 1)[0]
                    uttr_dic["spk"].append(sample_audio_item)

                # noise path
                noise_items = os.listdir(noise_path_item)
                uttr_dic["noise"] = noise_path_item + "/" + random.sample(noise_items, 1)[0]

                # snr
                uttr_dic["snr"] = random.uniform(0, 5)
                uttr_dic["noise_snr"] = random.uniform(5, 15)

                # room size
                room_length = random.uniform(4, 10)
                room_width = random.uniform(4, 10)
                room_height = random.uniform(2.5, 4)
                uttr_dic["room_size"] = [room_length, room_width, room_height]

                #distance beyween mic and sources
                uttr_dic["d"] = random.uniform(0.5, 2.0)

                # mic pos
                mic_z = 1.55
                mic_x = random.uniform(uttr_dic["d"], room_length-uttr_dic["d"])
                mic_y = random.uniform(uttr_dic["d"], room_width-uttr_dic["d"])
                assert (0<mic_x<room_length and 0<mic_y<room_width)
                uttr_dic["mic_pos"] = []
                mic_angle_list = [45, 315, 225, 135]
                for angle_item in range(4):
                    angle = mic_angle_list[angle_item]
                    mic_item_length = mic_x + np.cos(np.deg2rad(angle)) * mic_diameter/2
                    mic_item_width = mic_y - np.sin(np.deg2rad(angle)) * mic_diameter/2
                    assert (0<mic_item_length<room_length and 0<mic_item_width<room_width)
                    uttr_dic["mic_pos"].append([mic_item_length, mic_item_width, mic_z])

                #  source pos
                src_height = random.uniform(1.3, 1.8)
                uttr_dic["src_pos"] = []
                angles_list = []
                angles_list.append(random.randint(0, 359))
                deta = random.randint(30, 330)
                angle2 = angles_list[0] + deta
                if angle2 >= 360:
                    angle2 -= 360
                angles_list.append(angle2)
                uttr_dic["ang_list"] = angles_list
                for angle_list_item in angles_list:
                    source_length = mic_x + np.cos(np.deg2rad(angle_list_item)) * uttr_dic["d"]
                    source_width = mic_y - np.sin(np.deg2rad(angle_list_item)) * uttr_dic["d"]
                    assert (0<source_length<room_length and 0<source_width<room_width)
                    uttr_dic["src_pos"].append([source_length, source_width, src_height])
                    
                # noise pos
                noise_length = random.uniform(0, room_length)
                noise_width = random.uniform(0, room_width)
                noise_height = random.uniform(0, room_height)
                uttr_dic["nos_pos"] = [noise_length, noise_width, noise_height]
                
                # t60
                uttr_dic["rt60"] = random.uniform(0.2, 0.7)
                #  overlap ratio
                uttr_dic["overlap_ratio"] = random.uniform(0, 1)
                config.append(uttr_dic)
                
            pickle.dump(config, f)
            f.close()
            print("finish {} configs".format(item))


    
def create_mix(item, item_index, output_path, data_type, beam=True):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        spks = item["spk"]
        noise = item["noise"]
        spk1, _ = librosa.load(spks[0], fs)
        spk2, _ = librosa.load(spks[1], fs)
        noise, _ = librosa.load(noise, fs)
        
        overlap_ratio = item['overlap_ratio']
        actual_len = int(sig_len / (2 - overlap_ratio) * fs)
        overlap = int(actual_len*overlap_ratio)
        
        start_idx = 0
        end_idx = start_idx + actual_len

        spk1 = spk1[start_idx:end_idx]
        spk2 = spk2[start_idx:end_idx]
        def pad_length(x, length):
            if len(x) < length:
                zeros = np.zeros(length - len(x))
                return np.concatenate([x, zeros])
            else:
                return x[:length]
        spk1 = pad_length(spk1,actual_len)
        spk2 = pad_length(spk2,actual_len)

        # rescaling speaker and noise energy according to relative SNR
        spk1 = spk1 / np.sqrt(np.sum(spk1 ** 2) + 1e-8) * 1e2
        spk2 = spk2 / np.sqrt(np.sum(spk2 ** 2) + 1e-8) * 1e2
        spk2 = spk2 * np.power(10, item['snr'] / 20.)

        # repeat noise if necessary
        noise = noise[:int(sig_len * fs)]
        if len(noise) < int(sig_len * fs):
            num_repeat = int(sig_len * fs) // len(noise)
            res = int(sig_len * fs) - num_repeat * len(noise)
            noise = np.concatenate([np.concatenate([noise] * num_repeat), noise[:res]])

        noise = noise / np.sqrt(np.sum(noise ** 2) + 1e-8) * np.sqrt(np.sum((spk1 + spk2) ** 2) + 1e-8)
        noise = noise / np.power(10, item['noise_snr'] / 20.)

        mic_pos = np.asarray(item["mic_pos"])
        spk_pos = np.asarray(item["src_pos"])
        noise_pos = np.asarray(item["nos_pos"])
        room_size = np.asarray(item["room_size"])
        rt60 = item["rt60"]
        num_mic = len(mic_pos)

        # generate RIR  
        spk1_rir = rir.generate(c=c, fs=fs, r=mic_pos, s=spk_pos[0], L=room_size, reverberation_time=rt60,
                                nsample=num_samples)
        spk2_rir = rir.generate(c=c, fs=fs, r=mic_pos, s=spk_pos[1], L=room_size, reverberation_time=rt60,
                                nsample=num_samples)
        noise_rir = rir.generate(c=c, fs=fs, r=mic_pos, s=noise_pos, L=room_size, reverberation_time=rt60,
                                nsample=num_samples)

        # direct sound for model output
        spk1_rir1 = rir.generate(c=c, fs=fs, r=mic_pos, s=spk_pos[0], L=room_size, reverberation_time=0,
                                nsample=num_samples)
    
        spk1_rir2 = rir.generate(c=c, fs=fs, r=mic_pos, s=spk_pos[1], L=room_size, reverberation_time=0,
                        nsample=num_samples)    
 
        spk1_sig = []
        spk2_sig = []
        mix_sig = []

        for mic in range(num_mic):
            spk1_echoic_sig = signal.fftconvolve(spk1, spk1_rir[:,mic])
            spk2_echoic_sig = signal.fftconvolve(spk2, spk2_rir[:,mic])
            noise_echoic_sig = signal.fftconvolve(noise, noise_rir[:,mic])

            spk1_no_echoic_sig = signal.fftconvolve(spk1, spk1_rir1[:,mic])
            spk2_no_echoic_sig = signal.fftconvolve(spk2, spk1_rir2[:,mic])

            pad_length = int((1 - overlap_ratio) * actual_len)
            padding = np.zeros(pad_length)
            spk1_echoic_sig = np.concatenate([spk1_echoic_sig, padding])
            spk2_echoic_sig = np.concatenate([padding, spk2_echoic_sig])
            
            spk1_no_echoic_sig = np.concatenate([spk1_no_echoic_sig, padding])
            spk2_no_echoic_sig = np.concatenate([spk2_no_echoic_sig, padding])

            def pad_sig(x):
                if len(x) < sig_len*fs:
                    zeros = np.zeros(sig_len * fs - len(x))
                    return np.concatenate([x, zeros])
                else:
                    return x[:sig_len*fs]
                
            spk1_echoic_sig = pad_sig(spk1_echoic_sig)
            spk2_echoic_sig = pad_sig(spk2_echoic_sig)
            noise_echoic_sig = pad_sig(noise_echoic_sig)
            spk1_no_echoic_sig = pad_sig(spk1_no_echoic_sig)
            
            mixture = spk1_echoic_sig  + spk2_echoic_sig + noise_echoic_sig

            this_save_dir = output_path + "/" + data_type + "/" + "/" + "sample" + str(item_index + 1)
            if not os.path.exists(this_save_dir):
                os.makedirs(this_save_dir)
            mic_item_path = this_save_dir + "/" + str(item["ang_list"][0]) + "_" +str(item["ang_list"][1]) + "_mixture_mic_" + str(mic) + "_echo" + ".wav"
            sf.write(mic_item_path, mixture, fs)
            mic_item_path_n = this_save_dir + "/" + str(item["ang_list"][0]) + "_spk1_mic_" + str(mic) + "_no_echo" + ".wav"
            sf.write(mic_item_path_n, spk1_no_echoic_sig, fs)
            mic_item_path_n = this_save_dir + "/" + str(item["ang_list"][1]) + "_spk2_mic_" + str(mic) + "_no_echo" + ".wav"
            sf.write(mic_item_path_n, spk2_no_echoic_sig, fs)
            
            spk1_sig.append(spk1_no_echoic_sig)    
            spk2_sig.append(spk2_no_echoic_sig)     
            mix_sig.append(mixture)                 
            
        if beam:
            spk1_beam = np.concatenate(spk1_sig).reshape(num_mic, -1)
            spk2_beam = np.concatenate(spk2_sig).reshape(num_mic, -1)
            mix_beam = np.concatenate(mix_sig).reshape(num_mic, -1)
            
            spk1_beam = ds.run_ds(spk1_beam, item["ang_list"][0], fs)
            spk2_beam = ds.run_ds(spk2_beam, item["ang_list"][1], fs)
            mix_beam1 = ds.run_ds(mix_beam, item["ang_list"][0], fs)
            mix_beam2 = ds.run_ds(mix_beam, item["ang_list"][0], fs)
            
            beam_path = output_path + "/" + data_type + "/" + "sample" + str(item_index + 1)
            spk1_path = beam_path +  "/" + "spk1_beam_" + str(item["ang_list"][0]) + ".wav"
            mix1_path = beam_path + "/" + "mix1_beam_" + str(item["ang_list"][0]) + ".wav"
            spk2_path = beam_path +  "/" + "spk2_beam_" + str(item["ang_list"][1]) + ".wav"
            mix2_path = beam_path + "/" + "mix2_beam_" + str(item["ang_list"][1]) + ".wav"
            sf.write(spk1_path, spk1_beam, fs)
            sf.write(mix1_path, mix_beam1, fs)
            sf.write(spk2_path, spk2_beam, fs)
            sf.write(mix2_path, mix_beam2, fs)
        if item_index%5000==0:
            print("finish {}th train_data generation...".format(item_index))


def generation(data_type, config_path, output_path):
    
    pool = mp.Pool(processes=8)
    config_path = config_path + "/" + data_type + "_16khz.pkl"
    # load pickle file
    with open(config_path, "rb") as f:
        configs = pickle.load(f)
    # print(len(configs))
    for i in range(len(configs)):
        pool.apply_async(create_mix, (configs[i], i,  output_path, data_type))
    pool.close()
    pool.join()
    
def parse_pkl(scp_path):
    assert os.path.exists(scp_path)
    pkl_dict = {}
    with open(scp_path, 'rb') as f:
        configs = pickle.load(f)
        for i in range(len(configs)):
            pkl_dict[i] = configs[i]

    return pkl_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multi-channel Librispeech data')
    parser.add_argument('--task', metavar='task type', required=True, default=0,
                        help="Task type. 0: create config pkl. 1: generate data.")
    parser.add_argument('--config-path', metavar='absolute path', required=False, default='',
                        help="The path to config dictionary. Default is the current directory.")
    parser.add_argument('--libri-path', metavar='absolute path', required=True,
                        help="Absolute path for Librispeech folder containing train-clean-100, dev-clean and test-clean folders.")
    parser.add_argument('--noise-path', metavar='absolute path', required=True,
                        help="Absolute path for the 100 Nonspeech sound folder.")
    parser.add_argument('--output-path', metavar='absolute path', required=False, default='',
                        help="The path to the output directory. Default is the current directory.")
    args = parser.parse_args()
    if not args.task:
        create_mix_pickle(args.libri_path,args.noise_path, args.config_path)
    else:
        for type_i in ["train", "val", "test"]:
            generation(type_i,args.config_path,args.output_path)
