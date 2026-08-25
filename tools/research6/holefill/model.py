"""Hole-filler v1: stereo complex-STFT U-Net predicting a complex mask + additive
residual, identity at init (zero-init head). ~5-15M params, bf16-friendly.

out = (1 + m) * x + r   (complex, per channel)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 2048
HOP = 512
WIN = 2048


def stft(x):
    """x [B, 2, N] -> complex [B, 2, F, T]"""
    B, C, N = x.shape
    w = torch.hann_window(WIN, device=x.device, dtype=torch.float32)
    X = torch.stft(x.reshape(B * C, N).float(), N_FFT, HOP, WIN, w, center=True,
                   return_complex=True)
    _, Fb, T = X.shape
    return X.reshape(B, C, Fb, T)


def istft(X, n_samples):
    B, C, Fb, T = X.shape
    w = torch.hann_window(WIN, device=X.device, dtype=torch.float32)
    y = torch.istft(X.reshape(B * C, Fb, T), N_FFT, HOP, WIN, w, center=True,
                    length=n_samples)
    return y.reshape(B, C, n_samples)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g = max(1, ch // 16)
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class TimeGRU(nn.Module):
    """BiGRU over time, shared across frequency bins (channels = features)."""
    def __init__(self, ch, hidden):
        super().__init__()
        self.gru = nn.GRU(ch, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, ch)

    def forward(self, x):  # [B, C, F, T]
        B, C, Fb, T = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * Fb, T, C)
        h, _ = self.gru(h.float())
        h = self.proj(h)
        h = h.reshape(B, Fb, T, C).permute(0, 3, 1, 2)
        return x + h.to(x.dtype)


class HoleFiller(nn.Module):
    def __init__(self, widths=(48, 64, 96, 128, 192), strides=((2, 2), (2, 1), (2, 2), (2, 1), (2, 2)),
                 gru_hidden=96):
        super().__init__()
        self.strides = strides
        self.stem = nn.Conv2d(4, widths[0], 3, padding=1)
        enc, downs = [], []
        for i, s in enumerate(strides):
            wi = widths[i]
            wo = widths[min(i + 1, len(widths) - 1)]
            enc.append(nn.Sequential(ResBlock(wi), ResBlock(wi)))
            downs.append(nn.Conv2d(wi, wo, (4 if s[0] == 2 else 3, 4 if s[1] == 2 else 3),
                                   stride=s, padding=(1, 1)))
        self.enc = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)
        wb = widths[-1]
        self.mid = nn.Sequential(ResBlock(wb), ResBlock(wb))
        self.tgru = TimeGRU(wb, gru_hidden)
        ups, dec = [], []
        for i in reversed(range(len(strides))):
            wi = widths[min(i + 1, len(widths) - 1)]
            wo = widths[i]
            ups.append(nn.ConvTranspose2d(wi, wo, (4 if strides[i][0] == 2 else 3,
                                                   4 if strides[i][1] == 2 else 3),
                                          stride=strides[i], padding=(1, 1)))
            dec.append(nn.Sequential(nn.Conv2d(2 * wo, wo, 3, padding=1),
                                     ResBlock(wo)))
        self.ups = nn.ModuleList(ups)
        self.dec = nn.ModuleList(dec)
        self.head = nn.Conv2d(widths[0], 8, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def net(self, p):  # p [B, 4, F1024, T]
        h = self.stem(p)
        skips = []
        for blk, dn in zip(self.enc, self.downs):
            h = blk(h)
            skips.append(h)
            h = dn(h)
        h = self.mid(h)
        h = self.tgru(h)
        for up, de, sk in zip(self.ups, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != sk.shape[-2:]:
                h = h[..., :sk.shape[-2], :sk.shape[-1]]
            h = de(torch.cat([h, sk], dim=1))
        return self.head(h)

    def forward(self, x):
        """x [B, 2, N] normalized waveform -> y [B, 2, N]"""
        B, C, N = x.shape
        X = stft(x)                     # [B, 2, 1025, T] complex
        Xt = X[:, :, :1024, :]
        T = Xt.shape[-1]
        T8 = ((T + 7) // 8) * 8
        p = torch.cat([Xt.real, Xt.imag], dim=1)      # [B, 4, 1024, T]
        p = F.pad(p, (0, T8 - T))
        out = self.net(p)[..., :T].float()
        m_re, m_im = out[:, 0:2], out[:, 2:4]
        r_re, r_im = out[:, 4:6], out[:, 6:8]
        xr, xi = Xt.real, Xt.imag
        yr = xr * (1.0 + m_re) - xi * m_im + r_re
        yi = xi * (1.0 + m_re) + xr * m_im + r_im
        Y = torch.complex(yr, yi)
        Y = torch.cat([Y, X[:, :, 1024:, :]], dim=2)  # Nyquist passthrough
        return istft(Y, N)


class MRSTFTLoss(nn.Module):
    """Multi-res STFT magnitude L1 + power-compressed (c=0.3) magnitude L1 +
    small waveform L1 (Braun & Tashev compression; MSG-style fidelity-first)."""
    RES = ((512, 128), (1024, 256), (2048, 512))

    def __init__(self, w_mag=1.0, w_comp=2.0, w_wav=0.1, c=0.3):
        super().__init__()
        self.w_mag, self.w_comp, self.w_wav, self.c = w_mag, w_comp, w_wav, c

    def forward(self, y, t):
        """y, t [B, 2, N]"""
        B, C, N = y.shape
        yf = y.reshape(B * C, N).float()
        tf = t.reshape(B * C, N).float()
        l_mag = y.new_zeros(())
        l_comp = y.new_zeros(())
        for nfft, hop in self.RES:
            w = torch.hann_window(nfft, device=y.device)
            Ym = torch.stft(yf, nfft, hop, nfft, w, center=True, return_complex=True).abs()
            Tm = torch.stft(tf, nfft, hop, nfft, w, center=True, return_complex=True).abs()
            l_mag = l_mag + (Ym - Tm).abs().mean()
            l_comp = l_comp + ((Ym + 1e-8) ** self.c - (Tm + 1e-8) ** self.c).abs().mean()
        l_mag = l_mag / len(self.RES)
        l_comp = l_comp / len(self.RES)
        l_wav = (yf - tf).abs().mean()
        total = self.w_mag * l_mag + self.w_comp * l_comp + self.w_wav * l_wav
        return total, {"mag": l_mag.detach(), "comp": l_comp.detach(), "wav": l_wav.detach()}


def count_params(m):
    return sum(p.numel() for p in m.parameters())
