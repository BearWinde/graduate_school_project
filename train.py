"""
Train script.
"""
import numpy as np
from core import multiscale_fft, get_scheduler,mean_std_loudness
import torch
import yaml 
import shutil
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import datetime
from tensorboardX import SummaryWriter
import telegram
import asyncio

my_token = ''
my_chat_id = 

with open("config_violin.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
# duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]
hidden_size = config["train"]["hidden_size"]
n_harmonic = config["train"]["n_harmonic"]
n_bands = config["train"]["n_bands"]
n_wavetables = config["train"]["n_wavetables"]
n_mfcc = config["train"]["n_mfcc"]
train_lr = config["train"]["start_lr"]
epochs = config["train"]["epochs"]
spec_mode = config["train"]["spec_mode"]
n_mels = config["train"]["n_mels"]
model_name = config["train"]["model_name"]
effect = config["train"]["effect"]
pich_model = config["train"]["pich_model"]
print("""
======================
sr: {}
block_size: {}
batch_size: {}
scales: {}
overlap: {}
hidden_size: {}
n_harmonic: {}
n_bands: {}
n_wavetables: {}
n_mfcc: {}
train_lr: {}
======================
""".format(sr, block_size, batch_size, scales, overlap,
           hidden_size, n_harmonic, n_bands, n_wavetables, n_mfcc, train_lr))

model = WTS(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
            block_size=block_size, n_wavetables=n_wavetables, mode=model_name,
            n_mels=n_mels, spec_mode=spec_mode, n_mfcc=n_mfcc
            )
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=train_lr)

# 選擇是否要效果或是要哪個效果組件
effect = effect

# 選擇輸入要梅爾倒譜還是梅爾頻譜
if spec_mode == "mfcc":
    spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)

if spec_mode == "mel":
    spec = Spectrogram.MelSpectrogram(n_mels=n_mels)


train_dl = get_data_loader(config, mode="train", batch_size=batch_size)

# 計算資料集平均與標準差
mean_loudness, std_loudness = mean_std_loudness(train_dl)

# 進程管控
schedule = get_scheduler(
    len(train_dl),
    config["train"]["start_lr"],
    config["train"]["stop_lr"],
    config["train"]["decay_over"],
)

async def send(msg, chat_id, token=my_token):
    bot = telegram.Bot(token=token)
    await bot.sendMessage(chat_id=chat_id, text=msg)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/' + current_time +"_"+ model_name +"_spec_"+spec_mode+"_effect_"+ effect + "_pich_"+ pich_model +"_"+str(len(train_dl)) +'/train'

train_summary_writer = SummaryWriter(train_log_dir)
f1 = open("config_violin.yaml",'r')     # 開啟為可讀取
f2 = open(train_log_dir+"/config.yaml",'a')    # 開啟為可添加
shutil.copyfileobj(f1,f2)
idx = 0
for ep in tqdm(range(1, epochs + 1)):
    for audio_path, y, loudness, pitch ,duration_secs in tqdm(train_dl):
        mfcc = spec(y)
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
        loudness = (loudness - mean_loudness) / std_loudness

        mfcc = mfcc.cuda()
        pitch = pitch.cuda()
        loudness = loudness.cuda()

        output = model(mfcc, pitch, loudness,duration_secs,effect)
        
        ori_stft = multiscale_fft(
                    torch.tensor(y).squeeze(),
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
            lin_loss = (s_x - s_y).abs().mean()
            loss += lin_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_summary_writer.add_scalar('loss', loss.item(), global_step=idx)
        
        
        if idx % 1000 == 0:
            torch.save(model.state_dict(), 
                       "D:/wenzo/wave-synth/pt/model_" + 
                       model_name + 
                       "_spec_" + 
                       spec_mode + 
                       "_effect_" + 
                       effect +
                       "_pich_"+
                       pich_model +
                       "_dataset_"+
                       str(len(train_dl))+ 
                       "_step_" +str(idx)+".pt")
        
        idx += 1
        