from __future__ import annotations

import math
import torch
from torch import nn


class GatedCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.dim = dim

    def forward(self, rgb, frequency, return_attention=False):
        q = self.query(rgb).unsqueeze(1)
        k, v = self.key(frequency), self.value(frequency)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dim), dim=-1)
        attended = (weights @ v).squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([rgb, attended], dim=-1)))
        fused = gate * rgb + (1 - gate) * attended
        if return_attention:
            return fused, weights.squeeze(1), gate
        return fused


def _encoder(arch, pretrained, in_chans, checkpoint=None):
    try:
        import timm
    except ImportError as exc:
        raise ImportError('Full FPRA requires timm and a compatible torchvision. See requirements.txt.') from exc
    kwargs = dict(pretrained=pretrained and checkpoint is None, num_classes=0, in_chans=in_chans)
    if checkpoint:
        kwargs['checkpoint_path'] = str(checkpoint)
    encoder = timm.create_model(arch, **kwargs)
    return encoder, encoder.num_features


class FPRAClassifier(nn.Module):
    def __init__(self, fusion_dim=768, branches='both', pretrained=True, dropout=0.0,
                 clip_arch='vit_base_patch16_clip_224.openai',
                 swin_arch='swin_small_patch4_window7_224',
                 clip_checkpoint=None, swin_checkpoint=None, encoders=None):
        super().__init__()
        if branches not in {'rgb', 'amp', 'phase', 'both'}:
            raise ValueError('branches must be rgb, amp, phase, or both')
        if fusion_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid fusion dimension or dropout')
        self.branches = branches
        use_amp, use_phase = branches in {'amp', 'both'}, branches in {'phase', 'both'}
        if encoders is None:
            self.encoder_rgb, dim_rgb = _encoder(clip_arch, pretrained, 3, clip_checkpoint)
            if use_amp:
                self.encoder_amp, dim_amp = _encoder(swin_arch, pretrained, 1, swin_checkpoint)
            if use_phase:
                self.encoder_phase, dim_phase = _encoder(swin_arch, pretrained, 1, swin_checkpoint)
        else:
            self.encoder_rgb, dim_rgb = encoders['rgb']
            if use_amp:
                self.encoder_amp, dim_amp = encoders['amp']
            if use_phase:
                self.encoder_phase, dim_phase = encoders['phase']
        for parameter in self.encoder_rgb.parameters():
            parameter.requires_grad_(False)
        self.encoder_rgb.eval()
        self.proj_rgb = nn.Linear(dim_rgb, fusion_dim)
        if use_amp:
            self.proj_amp = nn.Linear(dim_amp, fusion_dim)
        if use_phase:
            self.proj_phase = nn.Linear(dim_phase, fusion_dim)
        if branches != 'rgb':
            self.fusion = GatedCrossAttention(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, 1))

    def train(self, mode=True):
        super().train(mode)

        self.encoder_rgb.eval()
        return self

    def forward_features(self, rgb, amp, phase, return_attention=False):
        with torch.no_grad():
            rgb = self.encoder_rgb(rgb)
        rgb = self.proj_rgb(rgb)
        frequency = []
        if self.branches in {'amp', 'both'}:
            frequency.append(self.proj_amp(self.encoder_amp(amp)))
        if self.branches in {'phase', 'both'}:
            frequency.append(self.proj_phase(self.encoder_phase(phase)))
        if not frequency:
            if return_attention:
                return rgb, None, None
            return rgb
        return self.fusion(rgb, torch.stack(frequency, dim=1), return_attention)

    def forward(self, rgb, amp, phase):
        return self.classifier(self.forward_features(rgb, amp, phase)).squeeze(-1)


TriStreamClassifier = FPRAClassifier


class _SmokeEncoder(nn.Module):
    def __init__(self, channels, dim=16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(channels, dim, 3, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.network(x)


def smoke_model(branches='both'):

    encoders = {name: (_SmokeEncoder(channels), 16)
                for name, channels in [('rgb', 3), ('amp', 1), ('phase', 1)]}
    return FPRAClassifier(fusion_dim=16, branches=branches, encoders=encoders)
