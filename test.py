"""
Test script.
"""
import numpy as np
from core import multiscale_fft,mean_std_loudness
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import soundfile as sf

with open("config_violin.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
batch_size = config["train"]["batch_size"]
scales = config["test"]["scales"]
overlap = config["test"]["overlap"]
n_mels = config["test"]["n_mels"]
spec_mode = config["test"]["spec_mode"]
n_mfcc = config["test"]["n_mfcc"]
effect = config["test"]["effect"]
model_name=config["test"]["model_name"]
model_name_path=config["test"]["model_name_path"]

model = WTS(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=sr,
            block_size=block_size, n_wavetables=10, mode=model_name, n_mels=n_mels, spec_mode=spec_mode, n_mfcc=n_mfcc)
model.cuda()
model_name_path = model_name_path

model.load_state_dict(torch.load(model_name_path))
effect = effect

if spec_mode == "mfcc":
    spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)
if spec_mode == "mel":
    spec = Spectrogram.MelSpectrogram(n_mels=n_mels)
test_dl = get_data_loader(config, mode="test", batch_size=batch_size)
mean_loudness, std_loudness = mean_std_loudness(test_dl)

idx = 0
for audio_path,y, loudness, pitch , duration_secs in tqdm(test_dl):
    mfcc = spec(y)
    pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
    loudness = (loudness - mean_loudness) / std_loudness


    mfcc = mfcc.cuda()
    pitch = pitch.cuda()
    loudness = loudness.cuda()

    output = model(mfcc, pitch, loudness,duration_secs,effect)
    
    ori_stft = multiscale_fft(
                y.squeeze(),
                scales,
                overlap,
            )
    rec_stft = multiscale_fft(
        output.squeeze(),
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        s_x = s_x.cuda()
        s_y = s_y.cuda()
        lin_loss = ((s_x - s_y).abs()).mean()
        loss += lin_loss

    print("Test Loss: {:.4}".format(loss.item()))
    audio = output.data.reshape(-1).detach().cpu().numpy()
    audio_path= str(audio_path).split('/')
    sf.write(audio_path[-1][:-7] +'.wav', audio, sr, 'PCM_24')
    idx += 1

