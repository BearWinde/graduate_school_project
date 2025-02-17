"""
Differentiable wavetable synthesis component.
"""
import torch
from torch import nn
import numpy as np
from utils import *
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
from core import upsample
import random


def wavetable_osc(wavetable, freq, sr):
    """
    General wavetable synthesis oscilator.
    """
    freq = freq.squeeze()
    increment = freq / sr * wavetable.shape[0]
    increment= torch.unsqueeze(increment,0)
    index = torch.cumsum(increment, dim=1) - increment[0]
    index = index % wavetable.shape[0]


    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()

    output = wavetable[index_low] + alpha * (wavetable[index_high % wavetable.shape[0]] - wavetable[index_low])
        
    return output


def generate_wavetable(length, f, cycle=1, phase=0):
    """
    Generate a wavetable of specified length using 
    function f(x) where x is phase.
    Period of f is assumed to be 2 pi.
    """
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(cycle * 2 * np.pi * i / length + 2 * phase * np.pi)
    return torch.tensor(wavetable)


class WavetableSynth(nn.Module):
    def __init__(self,
                 wavetables=None,
                 n_wavetables=64,
                 wavetable_len=512,
                 sr=16000,
                 block_size=160):
        super(WavetableSynth, self).__init__()
        if wavetables is None: 
            self.wavetables = []
            for _ in range(n_wavetables):
                cur = nn.Parameter(torch.empty(wavetable_len).normal_(mean=0, std=0.01))
                self.wavetables.append(cur)

            self.wavetables = nn.ParameterList(self.wavetables)

            for idx, wt in enumerate(self.wavetables):
                if idx == 0:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=1, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 1:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=2, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 2:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=3, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                elif idx == 3:
                    wt.data = generate_wavetable(wavetable_len, np.sin, cycle=4, phase=random.uniform(0, 1))
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = False
                else:
                    wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                    wt.requires_grad = True
            

        else:
            self.wavetables = wavetables
        
        self.sr = sr
        self.block_size = block_size
        self.attention_softmax = nn.Softmax(dim=0)
        self.n_wavetables = n_wavetables

    def forward(self, pitch, amplitude,sec):
        attention = nn.Parameter(torch.randn(self.n_wavetables, 100 * int(sec))).cuda()
        output_waveform_lst = []
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            if wt_idx not in [0, 1, 2, 3]:
                wt = nn.Tanh()(wt)  # ensure wavetable range is between [-1, 1]
            waveform = wavetable_osc(wt, pitch, self.sr)
            output_waveform_lst.append(waveform)

        # apply attention 
        attention = self.attention_softmax(attention)
        attention_upsample = upsample(attention.unsqueeze(-1), self.block_size).squeeze()

        output_waveform = torch.stack(output_waveform_lst, dim=1)
        output_waveform = output_waveform * attention_upsample
        output_waveform_after = torch.sum(output_waveform, dim=1)
      
        output_waveform_after = output_waveform_after.unsqueeze(-1)
        output_waveform_after = output_waveform_after * amplitude
       
        return output_waveform_after


"""
波表模塊測試
"""
if __name__ == "__main__":
    
    wavetable_len = 512
    sr = 16000
    duration = 4
    freq_t = [739.99 for _ in range(sr)] + [523.25 for _ in range(sr)] + [349.23 for _ in range(sr * 2)]
   
    freq_t = torch.tensor(freq_t)
    freq_t = torch.stack([freq_t, freq_t, freq_t], dim=0)
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    sine_wavetable =[sine_wavetable.tolist(),sine_wavetable.tolist()]
    wavetable = torch.tensor(sine_wavetable)
    
    wt_synth = WavetableSynth(wavetables=wavetable, sr=sr, duration_secs=4)
    amplitude_t = torch.ones(sr * duration,)
    amplitude_t = torch.stack([amplitude_t, amplitude_t, amplitude_t], dim=0)
    amplitude_t = amplitude_t.unsqueeze(-1)

    y = wt_synth(freq_t, amplitude_t)
    print(y.squeeze()[0].detach().numpy(), 'y')
    plt.plot(y.squeeze()[0].detach().numpy())
    plt.show()
    sf.write('test_3s_v1.wav', y.squeeze()[0].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v2.wav', y.squeeze()[1].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v3.wav', y.squeeze()[2].detach().numpy(), sr, 'PCM_24')



