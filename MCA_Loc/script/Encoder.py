import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class DilatedSeparableConv1d(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, bias=False):
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        # 1. Depthwise Conv with Dilation
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        # 2. Pointwise Conv
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ImprovedLightAttention(nn.Module):
    def __init__(self, emb_dim=128, dropout=0.1, kernel_size=15, num_layers=2, final_dim=300):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."

        self.fea_convs = nn.ModuleList([
            nn.Sequential(
                DilatedSeparableConv1d(emb_dim, emb_dim, kernel_size, dilation=2 ** i),
                nn.ReLU(),
                nn.BatchNorm1d(emb_dim)
            ) for i in range(num_layers)
        ])

        self.att_convs = nn.ModuleList([
            nn.Sequential(
                DilatedSeparableConv1d(emb_dim, emb_dim, kernel_size=5, dilation=1),
                nn.ReLU(),
                nn.BatchNorm1d(emb_dim)
            ) for i in range(num_layers)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.MLP = nn.Sequential(
            nn.Linear(2 * emb_dim, final_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(final_dim),
        )

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len, emb_dim]
            mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        """
        #  [B, D, L]
        x = x.transpose(1, 2)

        fea = x
        for conv_unit in self.fea_convs:
            fea = conv_unit(fea) + fea  # Residual connection
        fea_o = self.dropout(fea)
        att = x
        for conv_unit in self.att_convs:
            att = conv_unit(att)

        mask_expanded = mask.unsqueeze(1)
        att_masked = att.masked_fill(mask_expanded == 0, -1e9)
        att_weights = self.softmax(att_masked)  # [B, D, L]

        att_o = torch.sum(fea_o * att_weights, dim=-1)  # [B, D]

        fea_max = fea_o.masked_fill(mask_expanded == 0, -1e9)
        fea_o_max, _ = torch.max(fea_max, dim=-1)  # [B, D]
        out = torch.cat([att_o, fea_o_max], dim=-1)
        out = out.view(out.size(0), -1)
        return self.MLP(out), att_weights


class EncoderB(nn.Module):
    def __init__(self, go_dim=1024, sa_dim=1280, out_dim=256):

        self.go_proj = nn.Sequential(nn.Linear(go_dim, 512), nn.GELU())
        self.sa_proj = nn.Sequential(nn.Linear(sa_dim, 512), nn.GELU())


        self.gate_generator = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )


        self.bottleneck = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim)  
        )

    def forward(self, go_anno, sa_fea):
        g = self.go_proj(go_anno)
        s = self.sa_proj(sa_fea)

        g_refined = g * self.gate_generator(s)

        combined = torch.cat([g_refined, s], dim=-1)
        return self.bottleneck(combined)




