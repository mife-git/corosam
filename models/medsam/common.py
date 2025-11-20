# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Type


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class ConvAdapter(nn.Module):
    def __init__(self, D_features, skip_connect=True, channel_reduction=1):
        super().__init__()
        self.skip_connect = skip_connect
        self.D_hidden_features = int(D_features * channel_reduction)
        self.norm = nn.LayerNorm(D_features)

        self.conv1 = nn.Conv2d(in_channels=D_features, out_channels=self.D_hidden_features, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.ConvTranspose2d(in_channels=self.D_hidden_features, out_channels=D_features, kernel_size=3, padding=1)

    def forward(self, x):
        # Reshape to 4D tensor for Conv2d operations
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # L = H * W
        x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2)
        transformed = self.conv2(self.act(self.conv1(x_reshaped)))
        transformed = transformed.permute(0, 2, 3, 1).reshape(B, L, C)

        # Apply gating and residual
        if self.skip_connect:
            x = x + transformed
        else:
            x = transformed
        return self.norm(x)

class AdapterSAMMed(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x -> (B, L, C) -> (B, C, H, W)
        B, L, C = x.size()
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, L, C) -> (B, C, H, W)

        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        x_spatial = self.spatial(x_channel)

        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial

        # (B, C, H, W) -> (B, L, C)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return self.norm(x)


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
