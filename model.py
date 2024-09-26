"""
Diff-WTS model. Main adapted from https://github.com/acids-ircam/ddsp_pytorch.
"""
from core import harmonic_synth
from torchaudio.io import AudioEffector
from wavetable_synth import WavetableSynth
import torch
import torch.nn as nn
from core import mlp, gru, scale_function, remove_above_nyquist, upsample
from core import amp_to_impulse_response, fft_convolve,exp_sigmoid,variable_length_delay
import math
from torchvision.transforms import Resize
import torchaudio

class ModDelay(nn.Module):
  """可變長度延遲調製效果器模塊"""

  def __init__(self,
               center_ms=15.0,
               depth_ms=20.0,
               sample_rate=16000,
               gain_scale_fn=exp_sigmoid,
               phase_scale_fn=nn.Sigmoid(),
               add_dry=True,
               name='mod_delay'):
    super().__init__()
    
    self.center_ms = nn.Parameter(torch.tensor(float(center_ms)))
    self.depth_ms = nn.Parameter(torch.tensor(float(depth_ms)))
    self.sample_rate = sample_rate
    self.gain_scale_fn = exp_sigmoid
    self.phase_scale_fn = phase_scale_fn
    self.add_dry = add_dry

  def get_controls(self, audio, gain, phase):
    """Convert network outputs into magnitudes response.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Amplitude of modulated signal. Shape [batch_size, n_samples, 1].
      phase: Relative delay time. Shape [batch_size, n_samples, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    if self.gain_scale_fn is not None:
      gain = self.gain_scale_fn(gain)

    if self.phase_scale_fn is not None:
      phase = torch.sigmoid(phase)

  def forward(self, audio, gain, phase):
    """Filter audio with LTV-FIR filter.

    Args:
      audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
      gain: Amplitude of modulated signal. Shape [batch_size, n_samples, 1].
      phase: The normlaized instantaneous length of the delay, in the range of
        [center_ms - depth_ms, center_ms + depth_ms] from 0 to 1.0. Shape
        [batch_size, n_samples, 1].

    Returns:
      signal: Modulated audio of shape [batch, n_samples].
    """
    if self.gain_scale_fn is not None:
      gain = self.gain_scale_fn(gain)
    if self.phase_scale_fn is not None:
      phase = torch.sigmoid(phase)
    max_delay_ms = self.center_ms + self.depth_ms
    max_length_samples = int(self.sample_rate / 1000.0 * max_delay_ms)

    depth_phase = self.depth_ms / max_delay_ms
    center_phase = self.center_ms / max_delay_ms
    phase = phase * depth_phase + center_phase
    wet_audio = variable_length_delay(audio=audio,
                                           phase=phase,
                                           max_length=max_length_samples)

    if len(gain.shape) == 3:
      gain = gain[..., 0]

    wet_audio *= gain
    return (wet_audio + audio)
  
class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class WTS(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, n_wavetables, n_mfcc, n_mels,spec_mode="mfcc", mode="wavetable"):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.encoder = mlp(30, hidden_size, 3)
        #是否使用 波表模塊
        if mode != "wavetable":
          self.layer_norm = nn.LayerNorm(n_mfcc)
        # 根據輸入來選擇神經元層數
        if mode == "wavetable":
           if spec_mode =="mfcc":
              self.layer_norm = nn.LayerNorm(n_mfcc)
           else:
              self.layer_norm = nn.LayerNorm(n_mels)
        # 根據輸入來選擇神經元數量
        if mode != "wavetable" or spec_mode =="mfcc":
           self.gru_mfcc = nn.GRU(n_mfcc, 512, batch_first=True)
        else:
           self.gru_mfcc = nn.GRU(n_mels, 512, batch_first=True)

        self.mlp_mfcc = nn.Linear(512, 16)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3),
                                      mlp(1, hidden_size, 3),
                                      mlp(16, hidden_size, 3)])
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size * 4, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.moddelay = ModDelay()
        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.wts = WavetableSynth(n_wavetables=n_wavetables, 
                                  sr=sampling_rate, 
                                  block_size=block_size)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.mode = mode
        


    def forward(self, mfcc, pitch, loudness,duration_secs,effect):
        # encode mfcc first
        # use layer norm instead of trainable norm, not much difference found
        mfcc = self.layer_norm(torch.transpose(mfcc, 1, 2))
        mfcc = self.gru_mfcc(mfcc)[0]
        mfcc = self.mlp_mfcc(mfcc)
        # use image resize to align dimensions, ddsp also do this...
        mfcc = Resize(size=(int(duration_secs) * 100, 16))(mfcc)

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](mfcc)
        ], -1)
        
        hidden = torch.cat([self.gru(hidden)[0], hidden], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = self.proj_matrices[0](hidden)

        if self.mode != "wavetable":
            param = scale_function(self.proj_matrices[0](hidden))
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        total_amp = upsample(total_amp, self.block_size)    # TODO: wts can't backprop when using this total_amp, not sure why

        if self.mode == "wavetable":
            # diff-wave-synth synthesizer
            harmonic = self.wts(pitch, total_amp,duration_secs)
        else:
            harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic 

        # reverb part
        if effect == "reverb":
           signal = self.reverb(signal)
 
        if effect == "vibrato" or effect == "chorus" or effect == "flanger" :
          signal_1 = signal.squeeze(0).reshape(int(len(signal.squeeze(0))/self.sampling_rate),int(self.sampling_rate))
          gain = signal.squeeze(0).reshape(int(len(signal.squeeze(0))/self.sampling_rate),int(self.sampling_rate),1)
          signal=self.moddelay(signal_1,gain,gain)
          signal=signal.reshape(1,len(signal)*len(signal[0]))
        return signal