"""Log-mel front-end + CRNN per-frame voice-presence classifier (~2.5 M params)."""
import torch
import torch.nn as nn
import torchaudio

from common import SR, NFFT, WIN, HOP, NMELS


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=NFFT, win_length=WIN, hop_length=HOP,
            f_min=30.0, f_max=16000.0, n_mels=NMELS, power=2.0, center=True)

    def forward(self, wav):
        # wav [B, N] -> [B, NMELS, T]; per-segment CMVN
        m = self.mel(wav)
        x = torch.log(m + 1e-8)
        mu = x.mean(dim=2, keepdim=True)
        sd = x.std(dim=2, keepdim=True) + 1e-4
        return (x - mu) / sd


def conv_block(cin, cout, pool_f):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d((pool_f, 1)),
        nn.Dropout(0.1),
    )


class CRNN(nn.Module):
    def __init__(self, n_mels=NMELS, gru=192):
        super().__init__()
        self.frontend = LogMel()
        self.convs = nn.Sequential(
            conv_block(1, 32, 2),    # 128 -> 64
            conv_block(32, 64, 2),   # 64 -> 32
            conv_block(64, 128, 2),  # 32 -> 16
        )
        self.proj = nn.Linear(128 * (n_mels // 8), 2 * gru)
        self.gru = nn.GRU(2 * gru, gru, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * gru, 1)

    def forward(self, wav, specaug=False):
        x = self.frontend(wav)                    # [B, M, T]
        if specaug and self.training:
            B, M, T = x.shape
            for b in range(B):
                if torch.rand(1).item() < 0.5:
                    f0 = torch.randint(0, M - 16, (1,)).item()
                    x[b, f0: f0 + torch.randint(4, 16, (1,)).item()] = 0.0
                if torch.rand(1).item() < 0.5:
                    t0 = torch.randint(0, T - 40, (1,)).item()
                    x[b, :, t0: t0 + torch.randint(8, 40, (1,)).item()] = 0.0
        x = self.convs(x.unsqueeze(1))            # [B, C, M/8, T]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x = torch.relu(self.proj(x))
        x, _ = self.gru(x)
        return self.head(x).squeeze(-1)           # [B, T] logits


def count_params(m):
    return sum(p.numel() for p in m.parameters())
