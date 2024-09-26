import torchaudio
from tqdm import tqdm
import os
"""
批量切分音訊的範例
"""
list_train = os.listdir("D:/diff-wave-synth/violin/train/audio")
list_valid = os.listdir("D:/diff-wave-synth/violin/valid/audio")
list_test = os.listdir("D:/diff-wave-synth/violin/test/audio")

start_sec = 0
end_sec = 120
dict_train = {}
dict_valid = {}
dict_test = {}

for i in range(len(list_train)):
    dict_train[i]=list_train[i]

for i in range(len(list_valid)):
    dict_valid[i]=list_valid[i]

for i in range(len(list_test)):
    dict_test[i]=list_test[i]

for key in tqdm(dict_train):
    # 读取音频文件
    waveform, sample_rate = torchaudio.load('D:/diff-wave-synth/violin/train/audio/'+dict_train[key])
    # 定义剪辑的起始时间（以秒为单位）

    # 将时间转换为样本数
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    # 剪辑音频
    clipped_waveform = waveform[:, start_sample:end_sample]

    # 保存剪辑后的音频
    torchaudio.save('D:/diff-wave-synth/violin/train/'+ dict_train[key], clipped_waveform, sample_rate)
    
for key in tqdm(dict_valid):
    # 读取音频文件
    waveform, sample_rate = torchaudio.load('D:/diff-wave-synth/violin/valid/audio/'+dict_valid[key])
    # 定义剪辑的起始时间（以秒为单位）

    # 将时间转换为样本数
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    # 剪辑音频
    clipped_waveform = waveform[:, start_sample:end_sample]

    # 保存剪辑后的音频
    torchaudio.save('D:/diff-wave-synth/violin/valid/'+ dict_valid[key], clipped_waveform, sample_rate)

for key in tqdm(dict_test):
    # 读取音频文件
    waveform, sample_rate = torchaudio.load('D:/diff-wave-synth/violin/test/audio/'+dict_test[key])
    # 定义剪辑的起始时间（以秒为单位）

    # 将时间转换为样本数
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    # 剪辑音频
    clipped_waveform = waveform[:, start_sample:end_sample]
    clipped_waveform_1 = waveform[:, end_sample:end_sample*2]

    # 保存剪辑后的音频
    torchaudio.save('D:/diff-wave-synth/violin/test/1_'+ dict_test[key], clipped_waveform, sample_rate)
    torchaudio.save('D:/diff-wave-synth/violin/test/2_'+ dict_test[key], clipped_waveform_1, sample_rate)
