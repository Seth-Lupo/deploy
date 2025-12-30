"""
S3Gen: Speech Token to Waveform Generator
Adapted from Chatterbox TTS (Apache 2.0 License)
Modified from CosyVoice

Converts S3 speech tokens to audio waveforms using:
1. Token embedding
2. Upsampling encoder (Conformer)
3. Conditional Flow Matching (CFM) decoder
4. HiFT vocoder
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from .hifigan import HiFTGenerator, ConvRNNF0Predictor

logger = logging.getLogger(__name__)

# Constants
S3GEN_SR = 24000
S3GEN_SIL = 4299


class AttrDict(dict):
    """Dictionary with attribute access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# Default CFM parameters
CFM_PARAMS = AttrDict({
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
})


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding"""
    def __init__(self, in_channels, time_embed_dim, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class Block1D(nn.Module):
    """Basic 1D convolutional block"""
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(8, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        return self.block(x * mask) * mask


class ResnetBlock1D(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out)
        self.block2 = Block1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask, t):
        h = self.block1(x, mask)
        h = h + self.mlp(t).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class SimpleCFMDecoder(nn.Module):
    """Simplified CFM Decoder for inference only"""
    def __init__(self, in_channels=240, out_channels=80, channels=256, num_blocks=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_emb = SinusoidalPosEmb(in_channels)
        self.time_mlp = TimestepEmbedding(in_channels, channels * 4)

        # Input projection
        self.input_proj = nn.Conv1d(in_channels * 2 + 80 + 80, channels, 1)  # x + mu + spk + cond

        # Main blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResnetBlock1D(channels, channels, channels * 4))

        # Output projection
        self.output_proj = nn.Conv1d(channels, out_channels, 1)

    @property
    def dtype(self):
        return self.input_proj.weight.dtype

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        # Time embedding
        t_emb = self.time_emb(t).to(x.dtype)
        t_emb = self.time_mlp(t_emb)

        # Concatenate inputs
        inputs = [x, mu]
        if spks is not None:
            spks_expanded = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            inputs.append(spks_expanded)
        if cond is not None:
            inputs.append(cond)

        h = torch.cat(inputs, dim=1)
        h = self.input_proj(h)

        # Process through blocks
        for block in self.blocks:
            h = block(h, mask, t_emb)

        # Output
        out = self.output_proj(h * mask)
        return out * mask


class CausalConditionalCFM(nn.Module):
    """Conditional Flow Matching module"""
    def __init__(self, in_channels=240, cfm_params=None, n_spks=1, spk_emb_dim=80, estimator=None):
        super().__init__()
        if cfm_params is None:
            cfm_params = CFM_PARAMS

        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator or SimpleCFMDecoder(in_channels, 80)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None,
                noised_mels=None, meanflow=False):
        """
        Flow matching inference

        Args:
            mu: Conditioning [B, C, T]
            mask: Mask [B, 1, T]
            n_timesteps: Number of integration steps
            temperature: Noise temperature
            spks: Speaker embedding [B, D]
            cond: Additional conditioning [B, C, T]
            noised_mels: Pre-noised mels for continuation
            meanflow: Use meanflow scheduler
        """
        B = mu.size(0)
        z = torch.randn_like(mu) * temperature

        if noised_mels is not None:
            prompt_len = mu.size(2) - noised_mels.size(2)
            z[..., prompt_len:] = noised_mels

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if (not meanflow) and (self.t_scheduler == 'cosine'):
            t_span = 1 - torch.cos(t_span * 0.5 * np.pi)

        # Euler integration
        x = z
        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(0).expand(B)
            dxdt = self.estimator(x, mask, mu, t, spks, cond)
            dt = r - t[0]
            x = x + dt * dxdt

        return x, None


class SimpleUpsampleEncoder(nn.Module):
    """Simple upsampling encoder for token to mel resolution"""
    def __init__(self, input_size=512, output_size=80, upsample_rate=2):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.proj = nn.Linear(input_size, output_size * upsample_rate)
        self.output_dim = output_size

    def forward(self, x, x_len):
        # x: [B, T, C]
        h = self.proj(x)
        B, T, C = h.shape
        h = h.reshape(B, T * self.upsample_rate, C // self.upsample_rate)

        # Create mask
        mask = torch.ones(B, 1, T * self.upsample_rate, device=x.device, dtype=x.dtype)
        for i, l in enumerate(x_len):
            mask[i, :, l * self.upsample_rate:] = 0

        return h, mask

    def output_size(self):
        return self.output_dim


class S3Token2Wav(nn.Module):
    """
    S3 Speech Token to Waveform Generator

    Pipeline:
    1. Embed speech tokens
    2. Upsample to mel resolution
    3. Generate mel via CFM
    4. Vocoder mel to waveform
    """

    def __init__(
        self,
        vocab_size: int = 6561,
        hidden_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 80,
        upsample_rate: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.upsample_rate = upsample_rate

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Speaker embedding projection
        self.spk_embed_affine = nn.Linear(192, spk_embed_dim)  # CAMPPlus outputs 192-dim

        # Upsampling encoder
        self.encoder = SimpleUpsampleEncoder(hidden_size, output_size * 3, upsample_rate)
        self.encoder_proj = nn.Linear(output_size * 3, output_size)

        # CFM Decoder
        self.decoder = CausalConditionalCFM(
            in_channels=output_size * 3,
            n_spks=1,
            spk_emb_dim=spk_embed_dim,
        )

        # Vocoder
        self.f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=output_size)
        self.vocoder = HiFTGenerator(
            in_channels=output_size,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=S3GEN_SR,
            f0_predictor=self.f0_predictor,
        )

        # Fade-in for reducing artifacts
        self.fade_in_len = 24

    def _create_mask(self, lengths, max_len, device):
        """Create padding mask"""
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.float().unsqueeze(1)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: Optional[Dict] = None,
        n_cfm_timesteps: int = 10,
        finalize: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate waveform from speech tokens

        Args:
            speech_tokens: Token IDs [T] or [B, T]
            ref_dict: Reference audio dict with 'embedding', 'mel', etc.
            n_cfm_timesteps: CFM integration steps (2 for turbo)
            finalize: Whether to apply fade-in

        Returns:
            wav: Audio waveform [B, samples]
            mel: Generated mel-spectrogram [B, mel_channels, T]
        """
        # Handle input shape
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        B, T = speech_tokens.shape
        device = speech_tokens.device

        # Embed tokens
        token_len = torch.tensor([T], device=device)
        mask = torch.ones(B, 1, T, device=device, dtype=torch.float32)
        token_emb = self.token_embedding(speech_tokens) * mask.transpose(1, 2)

        # Upsample
        h, h_mask = self.encoder(token_emb, token_len)

        # Get speaker embedding
        if ref_dict is not None and 'embedding' in ref_dict:
            spk_emb = ref_dict['embedding']
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            spk_emb = F.normalize(spk_emb, dim=1)
            spk_emb = self.spk_embed_affine(spk_emb)
        else:
            spk_emb = torch.zeros(B, 80, device=device)

        # Conditioning from reference mel
        mel_len = h.shape[1]
        if ref_dict is not None and 'mel' in ref_dict:
            ref_mel = ref_dict['mel']
            if ref_mel.shape[-1] < mel_len:
                cond = F.pad(ref_mel, (0, mel_len - ref_mel.shape[-1]))
            else:
                cond = ref_mel[..., :mel_len]
        else:
            cond = torch.zeros(B, self.output_size, mel_len, device=device)

        # CFM generate mel
        h = self.encoder_proj(h)
        mel, _ = self.decoder(
            mu=h.transpose(1, 2),
            mask=h_mask,
            spks=spk_emb,
            cond=cond,
            n_timesteps=n_cfm_timesteps,
        )

        # Vocoder
        wav = self.vocoder(mel)

        # Apply fade-in to reduce artifacts
        if finalize and self.fade_in_len > 0:
            fade_len = min(self.fade_in_len * (S3GEN_SR // 1000), wav.shape[-1])
            fade_in = torch.linspace(0, 1, fade_len, device=device)
            wav[..., :fade_len] = wav[..., :fade_len] * fade_in

        return wav, mel

    @classmethod
    def from_pretrained(cls, weights_path: str, device: str = "cuda"):
        """Load model from safetensors weights"""
        from safetensors.torch import load_file

        model = cls()

        # Load weights
        state_dict = load_file(weights_path)

        # Map weight names if needed
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()

        return model
