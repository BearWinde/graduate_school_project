"""
Core functions. 
The code mainly comes from https://github.com/acids-ircam/ddsp_pytorch with minor adaptations.
"""
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
# from torchcrepeV2 import TorchCrepePredictor
import math
import tensorflow as tf

# torchcrepeV2 is my own version of crepe in torch, not released yet
# crepe_predictor = TorchCrepePredictor()


def safe_log(x):
    return torch.log(x + 1e-7)


@torch.no_grad()

# Utility Functions ------------------------------------------------------------
def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32).cpu()
  
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.

  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  m=nn.Sigmoid()
  return max_value *  m(x)**math.log(exponent) + threshold

def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l, _ ,_ in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def variable_length_delay(phase,
                          audio,
                          max_length: int = 128) :
  """Delay audio by a time-vaying amount using linear interpolation.

  Useful for modulation effects such as vibrato, chorus, and flanging.
  Args:
    phase: The normlaized instantaneous length of the delay, ranging from 0 to
      1.0. This corresponds to a delay of 0 to max_length samples. Shape
      [batch_size, n_samples, 1].
    audio: Audio signal to be delayed. Shape [batch_size, n_samples].
    max_length: Maximimum delay in samples.

  Returns:
    The delayed audio signal. Shape [batch_size, n_samples].
  """
  # Make causal by zero-padding audio up front.
  audio=nn.functional.pad(audio, [max_length - 1, 0, 0,0])

  # Cut audio up into frames of max_length.
  frames=audio.unfold(-1, max_length, 1)

  # Reverse frames so that [0, 1] phase corresponds to [0, max_length] delay.
  frames=torch.from_numpy(frames.data.cpu().numpy()[..., ::-1].copy())
  # Read audio from the past frames.
  return linear_lookup(phase, frames)

def linear_lookup(phase,
                  wavetables):
  """Lookup from wavetables with linear interpolation.

  Args:
    phase: The instantaneous phase of the base oscillator, ranging from 0 to
      1.0. This gives the position to lookup in the wavetable.
      Shape [batch_size, n_samples, 1].
    wavetables: Wavetables to be read from on lookup. Shape [batch_size,
      n_samples, n_wavetable] or [batch_size, n_wavetable].

  Returns:
    The resulting audio from linearly interpolated lookup of the wavetables at
      each point in time. Shape [batch_size, n_samples].
  """

  # Add a time dimension if not present.
  if len(wavetables.shape) == 2:
    wavetables = wavetables[:, tf.newaxis, :]

  # Add a wavetable dimension if not present.
  if len(phase.shape) == 2:
    phase = phase[:, :, tf.newaxis]

  # Add first sample to end of wavetable for smooth linear interpolation
  # between the last point in the wavetable and the first point.
  wavetables = torch.cat([wavetables, wavetables[..., 0:1]], axis=-1)
  n_wavetable = int(wavetables.shape[-1])

  # Get a phase value for each point on the wavetable.
  phase_wavetables = torch.linspace(0.0, 1.0, n_wavetable)
  # Get pair-wise distances from the oscillator phase to each wavetable point.
  # Axes are [batch, time, n_wavetable].
  phase_distance =(phase - phase_wavetables[tf.newaxis, tf.newaxis, :].cuda()).abs()

  # Put distance in units of wavetable samples.
  phase_distance *= n_wavetable - 1
  m = nn.ReLU()
  # Weighting for interpolation.
  # Distance is > 1.0 (and thus weights are 0.0) for all but nearest neighbors.
#   weights = tf.nn.relu(1.0 - phase_distance.cpu())
  weights = m(1.0 - phase_distance)
  weighted_wavetables = weights * wavetables.cuda()

  # Interpolated audio from summing the weighted wavetable at each timestep.
  return torch.sum(weighted_wavetables, axis=-1)

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def amplitude_to_db(amplitude):
    amin = 1e-20  # Avoid log(0) instabilities.
    db = torch.log10(torch.clamp(amplitude, min=amin))
    db *= 20.0
    return db


def extract_loudness(audio, sampling_rate, block_size=None, n_fft=2048, frame_rate=None):
    assert (block_size is None) != (frame_rate is None), "Specify exactly one of block_size or frame_rate"

    if frame_rate is not None:
        block_size = sampling_rate // frame_rate
    else:
        frame_rate = int(sampling_rate / block_size)

    if sampling_rate % frame_rate != 0:
        raise ValueError(
            'frame_rate: {} must evenly divide sample_rate: {}.'
            'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
            .format(frame_rate, sampling_rate))

    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio)

    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    audio = audio[None, :] if is_1d else audio

    # Take STFT.
    overlap = 1 - block_size / n_fft
    amplitude = torch.stft(audio, n_fft=n_fft, hop_length=block_size, center=True, pad_mode='reflect', return_complex=True).abs()
    amplitude = amplitude[:, :, :-1]
    
    # Compute power.
    power_db = amplitude_to_db(amplitude)

    # Perceptual weighting.
    frequencies = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weighting = li.A_weighting(frequencies)[None,:,None]
    loudness = power_db + a_weighting

    loudness = torch.mean(torch.pow(10, loudness / 10.0), axis=1)
    loudness = 10.0 * torch.log10(torch.clamp(loudness, min=1e-20))

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness
    loudness = loudness.numpy()
    # print(lou)
    return loudness


def extract_pitch(signal, sampling_rate, block_size, model_capacity="full"):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
        model_capacity="full"
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


# torchcrepeV2 is my own version of crepe in torch, not released yet
# def extract_pitch_v2(signal, sampling_rate, block_size, model_capacity="full"):
#     length = signal.shape[-1] // block_size
#     f0 = crepe_predictor.predict(
#         signal,
#         sampling_rate,
#         step_size=int(1000 * block_size / sampling_rate),
#         verbose=1,
#         center=True,
#         viterbi=True
#     )
#     if f0.shape[-1] != length:
#         f0 = np.interp(
#             np.linspace(0, 1, length, endpoint=False),
#             np.linspace(0, 1, f0.shape[-1], endpoint=False),
#             f0,
#         )

#     return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def get_scheduler(len_dataset, start_lr, stop_lr, length):
    def schedule(epoch):
        step = epoch * len_dataset
        if step < length:
            t = step / length
            return start_lr * (1 - t) + stop_lr * t
        else:
            return stop_lr

    return schedule