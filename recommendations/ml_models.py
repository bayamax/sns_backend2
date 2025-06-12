# recommendations/ml_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: 現在はgenerate_recommendations.py内で直接モデル定義を行っているため、
# このファイルは将来的な拡張用として保持。
# 実際の推論処理では generate_recommendations.py 内のモデル定義を使用。

class FollowPredictor(nn.Module):
    """ultimate_predict.pyと同等のシンプルなフォロー予測モデル"""
    def __init__(self, acc_dim=128, hid=256):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
        self.te = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hid*3, hid), nn.GELU(),
            nn.Linear(hid, hid//2), nn.GELU(),
            nn.Linear(hid//2, 1), nn.Sigmoid()
        )

    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)