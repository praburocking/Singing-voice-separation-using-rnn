#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import librosa as lb


def get_audio_file_data(input_dir, audio_file: str) -> np.ndarray:
    return lb.load(input_dir + audio_file, sr=None)


def extract_stft(data=None, sr=44100, time_win=20, n_fft=2048) -> np.ndarray:
    win_len = int((sr / 1000) * time_win)
    hop_len = int(win_len / 2)

    stft_out = lb.stft(data, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
    pow_out = np.abs(stft_out) ** 2
    return stft_out


def extract_mel_band_energies(stft, sr=44100, time_win=20, n_fft=2048, n_mels=40) -> np.ndarray:
    win_len = int((sr / 1000) * time_win)
    hop_len = int(win_len / 2)

    pow_out = np.abs(stft) ** 2
    mel_band = lb.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(mel_band, pow_out)
    return mel_energy


def create_feature(data, sr):
    stft = extract_stft(data=data,sr=sr)
    return stft

def create_features(data_dir,feature_dir,length,feature_prefix='',filter_folder=None):

    for cur_dir in os.listdir(data_dir):

        #if filter_folder==None or cur_dir.__contains__(filter_folder):
            for file in os.listdir(data_dir + cur_dir):
                cur_file=data_dir + cur_dir+'/'+file
                data,sr=lb.load(cur_file,sr=None)
                data=create_feature(data,sr)
                print(data.shape)
                cur_length= data.shape[1] if length==-1 else length
                print(cur_length)
                for j,i in enumerate(range(0,data.shape[1],cur_length)):
                    if i + cur_length > data.shape[1]:
                        print("breaking")
                        break

                    temp_cur_dir=feature_prefix + cur_dir.replace(" ", "_").replace("_", "") + "_seq_" + str(j) + '.npy'
                    print(temp_cur_dir+" "+cur_dir)
                    with open(feature_dir+temp_cur_dir, 'wb') as f:
                        temp_data=data[:,i:i+cur_length]
                        if(not temp_data.shape==(1025,cur_length)):
                            raise Exception("exception found size of temp_Data "+str(temp_data.shape)+" iteration "+str(i))
                        np.save(f,np.reshape(temp_data,(length,temp_data.shape[0])))
                       # break
                #break
            #break


def main():
    '''
        data and feature Directories
    '''
    audio_root_dir = '../data/raw_data/'
    sources_dir = audio_root_dir + 'Sources/'
    mix_dir = audio_root_dir + 'Mixtures/'
    feature_root_dir = '../data/features/'
    length=60
    create_features(mix_dir+'training/', feature_root_dir+'mix/train/', length, feature_prefix='mix_train_')
    create_features(mix_dir + 'testing/', feature_root_dir + 'mix/test/', -1, feature_prefix='mix_test_')
    create_features(sources_dir + 'testing/', feature_root_dir + 'source/test/', -1, feature_prefix='sou_test_')
    create_features(sources_dir + 'training/', feature_root_dir + 'source/train/', length, feature_prefix='sou_train_')



if __name__ == "__main__":
    main()

# EOF
