"""
HiFT Generator Vocoder
Adapted from CosyVoice (Apache 2.0 License)
Mel-spectrogram to waveform synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
import numpy as np
from typing import Optional


class Snake(nn.Module):
    """Snake activation function: x + 1/a * sin^2(ax)"""
    def __init__(self, channels, alpha=1.0, alpha_trainable=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels) * alpha, requires_grad=alpha_trainable)

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions and Snake activation"""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations1 = nn.ModuleList()
        self.activations2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=self.get_padding(kernel_size, d)
            )))
            self.convs2.append(weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=self.get_padding(kernel_size, 1)
            )))
            self.activations1.append(Snake(channels))
            self.activations2.append(Snake(channels))

    def get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.activations1, self.activations2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class SineGen(nn.Module):
    """Sine waveform generator with harmonics"""
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1,
                 noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            fn = torch.multiply(f0, torch.arange(1, self.harmonic_num + 2, device=f0.device).reshape(1, 1, -1))
            rad = fn / self.sampling_rate
            rad_values = torch.cumsum(rad, dim=1) % 1
            rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=f0.device)
            rand_ini = rand_ini.unsqueeze(1)
            rad_values = rad_values + rand_ini
            sines = torch.sin(2 * np.pi * rad_values)
            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sines)
            sines = sines * uv + noise
        return sines


class SourceModuleHnNSF(nn.Module):
    """Source module combining harmonic and noise sources"""
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class HiFTGenerator(nn.Module):
    """
    HiFT Generator: Mel-spectrogram to waveform vocoder
    Uses STFT/iSTFT with source filtering
    """
    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: list = [8, 8],
        upsample_kernel_sizes: list = [16, 16],
        istft_params: dict = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: list = [7, 11],
        source_resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: nn.Module = None,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_predictor = f0_predictor
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold
        )
        self.upsample_rates = upsample_rates

        # Pre-convolution
        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                base_channels // (2 ** i),
                base_channels // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2
            )))

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Source network
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            self.source_downs.append(weight_norm(Conv1d(
                1, ch, upsample_kernel_sizes[i] * 2,
                upsample_rates[i], padding=upsample_kernel_sizes[i] // 2
            )))
            for j, (k, d) in enumerate(zip(source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
                self.source_resblocks.append(ResBlock(ch, k, d))

        # STFT parameters
        self.n_fft = istft_params["n_fft"]
        self.hop_len = istft_params["hop_len"]
        self.istft_pre = weight_norm(Conv1d(base_channels // (2 ** self.num_upsamples), self.n_fft + 2, 7, 1, padding=3))

        # Post-convolution for STFT
        self.activation_post = Snake(self.n_fft + 2)
        self.conv_post = weight_norm(Conv1d(self.n_fft + 2, self.n_fft + 2, 7, 1, padding=3))

        self.audio_limit = audio_limit
        self.lrelu_slope = lrelu_slope

        # Register STFT window
        self.register_buffer('stft_window', torch.hann_window(self.n_fft))

    def forward(self, mel: torch.Tensor, f0: Optional[torch.Tensor] = None):
        """
        Args:
            mel: Mel-spectrogram [B, mel_channels, T]
            f0: F0 values [B, T] (optional, will predict if not provided)
        Returns:
            audio: Waveform [B, T * hop_product]
        """
        # Predict F0 if not provided
        if f0 is None:
            f0 = self.f0_predictor(mel)

        # Generate source signal
        upp = int(np.prod(self.upsample_rates))
        f0 = F.interpolate(f0.unsqueeze(1), scale_factor=upp, mode='linear', align_corners=False).squeeze(1)
        har_source = self.m_source(f0, upp).transpose(1, 2)

        # Pre-convolution
        x = self.conv_pre(mel)

        # Upsampling with residual blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            # Add source signal
            x_source = self.source_downs[i](har_source)
            for j in range(len(self.source_resblocks) // self.num_upsamples):
                x_source = self.source_resblocks[i * (len(self.source_resblocks) // self.num_upsamples) + j](x_source)
            x = x + x_source

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # STFT output
        x = self.istft_pre(x)
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Split magnitude and phase
        spec = torch.exp(x[:, :self.n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.n_fft // 2 + 1:, :])

        # iSTFT
        spec = spec.transpose(1, 2)
        phase = phase.transpose(1, 2)
        spec_complex = spec * torch.exp(1j * phase * np.pi)
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=self.stft_window,
            center=True
        )

        # Limit audio
        audio = torch.clamp(audio, -self.audio_limit, self.audio_limit)
        return audio

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        for down in self.source_downs:
            remove_weight_norm(down)
        for block in self.source_resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.istft_pre)
        remove_weight_norm(self.conv_post)


class ConvRNNF0Predictor(nn.Module):
    """F0 Predictor using convolutional layers"""
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))
