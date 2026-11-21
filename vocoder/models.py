"""HiFi-GAN (V2 generator configuration) — original implementation for the
tiny-voca-ja vocoder pilot, following Kong et al. 2020.

Generator: ~0.92M params with upsample_initial_channel=128.
Discriminators: Multi-Period (periods 2,3,5,7,11) + Multi-Scale.
Only the generator is needed at inference / for distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                  dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilations)
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                  dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilations)
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in list(self.convs1) + list(self.convs2):
            remove_weight_norm(c)


class Generator(nn.Module):
    def __init__(self, num_mels=80, upsample_rates=(8, 8, 2, 2),
                 upsample_kernel_sizes=(16, 16, 4, 4),
                 upsample_initial_channel=128,
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=(1, 3, 5)):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = weight_norm(nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for rate, k in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch, ch // 2, k, rate, padding=(k - rate) // 2)))
            ch //= 2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k in resblock_kernel_sizes:
                self.resblocks.append(ResBlock1(ch, k, resblock_dilations))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        channels = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.convs.append(weight_norm(nn.Conv2d(
                in_ch, out_ch, (kernel_size, 1), (stride, 1),
                padding=(get_padding(kernel_size, 1), 0))))
            in_ch = out_ch
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # x: [B, 1, T] -> [B, 1, T/p, p]
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, y, y_hat):
        outs_r, outs_g, fmaps_r, fmaps_g = [], [], [], []
        for d in self.discriminators:
            out_r, f_r = d(y)
            out_g, f_g = d(y_hat)
            outs_r.append(out_r)
            outs_g.append(out_g)
            fmaps_r.append(f_r)
            fmaps_g.append(f_g)
        return outs_r, outs_g, fmaps_r, fmaps_g


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            spectral_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            spectral_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            spectral_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            spectral_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            spectral_norm(nn.Conv1d(1024, 1024, 41, 1, padding=20)),
            spectral_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
        ])

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        outs_r, outs_g, fmaps_r, fmaps_g = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            out_r, f_r = d(y)
            out_g, f_g = d(y_hat)
            outs_r.append(out_r)
            outs_g.append(out_g)
            fmaps_r.append(f_r)
            fmaps_g.append(f_g)
        return outs_r, outs_g, fmaps_r, fmaps_g


def feature_loss(fmaps_r, fmaps_g):
    loss = 0.0
    for dr, dg in zip(fmaps_r, fmaps_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(outs_r, outs_g):
    loss = 0.0
    for dr, dg in zip(outs_r, outs_g):
        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
    return loss


def generator_loss(outs_g):
    loss = 0.0
    for dg in outs_g:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss
