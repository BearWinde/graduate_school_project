from pydub import AudioSegment
import torchaudio
import torch
import numpy as np
import os
import soundfile as sf
from torchaudio.io import AudioEffector
from noisereduce.torchgate import TorchGate as TG
import scipy.io.wavfile as wav
import scipy.signal as signal
import librosa 
import matplotlib.pyplot as plt

## 增加音訊長度
def create_audio(input_path,output_path,start_time,end_time):
    # 加载音频文件
    audio1 = AudioSegment.from_wav(input_path)

    # 剪辑音频文件
    start_time = start_time  # 开始时间（毫秒）
    end_time = end_time    # 结束时间（毫秒）
    clipped_audio1 = audio1[start_time:end_time]
    clipped_audio2 = audio1[0:10]
    # 串接音频文件
    concatenated_audio = clipped_audio1 + clipped_audio2
    # 保存结果
    concatenated_audio.export(output_path, format="wav")

## 切分音訊
def cut_audio(input_path,output_path,start_time,end_time):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(input_path)
    # 将时间转换为样本数
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    # 剪辑音频
    clipped_waveform = waveform[:, start_sample:end_sample]

    # 保存剪辑后的音频
    torchaudio.save(output_path, clipped_waveform, sample_rate)

## mp3 轉 WAV
def mp3_wav(mp3_path,wav_path,path):
    song = AudioSegment.from_mp3(mp3_path)
    song.export(wav_path+"_"+path,format="wav")

## 效果器
def show(effect, path, stereo=False):
    waveform, sample_rate = torchaudio.load(path,channels_first=False)
    wf = torch.cat([waveform] * 2, dim=1) if stereo else waveform
    effector = AudioEffector(effect=effect, pad_end=False)
    result = effector.apply(wf, int(sample_rate))
    return result,sample_rate

"""
批量套件添加音訊效果的範例
"""
# transform wav path_fold
path_piano ="D:/wenzo/wave-synth/violin/violin_48/audio"

# transform before save wave path_fold
save_effect_path ="D:/wenzo/wave-synth/violin/violin_48_effect_new/audio/"
list_train = os.listdir(path_piano)

effect = "tremolo=f=8:d=0.8" #效果器編號
for path in list_train:
    print(path)
    load_path = str(path_piano)+"/"+str(path)
    audio,sample_rate =show(effect,load_path)
    sf.write(save_effect_path+path,  audio.data.reshape(-1).detach().cpu().numpy(), sample_rate)


## 雙聲道切分
def two_channel(path,output_left,output_right):
    sound1 = AudioSegment.from_file(path, format="wav")
    print(sound1.channels)
    if sound1.channels==2:
        channels=sound1.split_to_mono()
        channels[0].export(output_left, codec="pcm_s16le",format="wav")
        channels[1].export(output_right, codec="pcm_s16le", format="wav")
        print("done!")
    else:
        print("channel is not eqaul 2")

## 採樣率轉換
def resample(origin_wav_path,resample_wav_path,new_sample_rate):
    audio_data ,sample_rate= librosa.load(origin_wav_path)
    resample_audio=librosa.resample(audio_data,sample_rate,new_sample_rate)
    wav.write(resample_wav_path,new_sample_rate,resample_audio)


def overlay(overlay_path_1,overlay_path_2,output_path):
    T1=AudioSegment.from_wav(overlay_path_1)
    T2=AudioSegment.from_wav(overlay_path_2)
    output=T1.overlay(T2)
    output = AudioSegment.from_mono_audiosegments(T1,T2)
    output.export(overlay_path_2,format="wav")

#  mel轉圖(原採樣率、原通道數)
def mel_spectrogram_image(file_path):
    audio, fs = librosa.load(
        file_path,
        sr=None,
        mono=True
    )
    L = len(audio)
    frame_length = 0.025
    print("time", L / fs)
    frame_size = int(fs * frame_length)
    print("nfft:",frame_size)

    mel_spec =librosa.feature.melspectrogram(
        audio,
        sr = fs,
        n_fft = 1024
        )
    mel_spec = librosa.power_to_db(
        mel_spec,
        ref = np.max
        )
    librosa.display.specshow(
        mel_spec,
        sr=fs,
        x_axis="time",
        y_axis="mel"
        )
    plt.ylabel("Mel Frequency")
    plt.xlabel("time(s)")
    fname = os.path.basename(file_path)
    plt.title(str(fname)+"(Mel Spectrogram)")
    plt.savefig("{0}-mel_specgram.png".format(str(fname)))
    plt.show()

#  語譜圖
def specgram_img(file_path):
    audio, fs = librosa.load(
        file_path,
        sr=None,
        mono=True
    )    
    
    L = len(audio)
    frame_length = 0.025
    print("time", L / fs)
    frame_size = fs * frame_length
    print("nfft:",frame_size)

    plt.specgram(
        audio, 
        Fs=fs, 
        NFFT=1024, 
        noverlap=256)
    fname = os.path.basename(file_path)
    plt.ylabel("Specgram Frequency")
    plt.xlabel("time(s)")
    plt.title(fname + "-specgram")
    plt.savefig("{0}-specgram.png".format(fname))
    plt.show()