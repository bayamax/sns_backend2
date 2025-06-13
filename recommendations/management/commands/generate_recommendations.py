import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────
# ▼▼▼ ここだけ編集すれば OK ▼▼▼
OPENAI_API_KEY = ""   # ここに sk-xxxx を貼る／空なら環境変数を使用
# ▲▲▲ ここだけ編集すれば OK ▲▲▲
# ──────────────────────────────────────────────

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    print("[WARNING] OPENAI_API_KEY が空です。投稿埋め込みは実行されません。")

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from posts.models import Post
from recommendations.models import UserEmbedding, UserRecommendation

User = get_user_model()

NODE2VEC_DIM = 128
OPENAI_DIM = 3072
MAX_POSTS = 50

# =========================  モデル定義  =========================
class AttentionPooling(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, acc_dim=NODE2VEC_DIM, n_head=8):
        super().__init__()
        self.mha  = nn.MultiheadAttention(post_dim, n_head, batch_first=True, dropout=0.1)
        self.ln   = nn.LayerNorm(post_dim)
        self.proj = nn.Sequential(
            nn.Linear(post_dim, acc_dim*2), nn.GELU(),
            nn.Linear(acc_dim*2, acc_dim)
        )
        self.sc   = nn.Linear(post_dim, 1, bias=False)

    def forward(self, x, mask=None):
        x  = F.normalize(x, p=2, dim=-1)
        att,_ = self.mha(x, x, x, key_padding_mask=(mask==0) if mask is not None else None)
        x  = self.ln(x + att)
        scr = self.sc(x).squeeze(-1)
        if mask is not None:
            scr = scr.masked_fill(mask == 0, -1e9)
        w   = F.softmax(scr, dim=-1)
        acc = torch.sum(x * w.unsqueeze(-1), dim=1)
        return F.normalize(self.proj(acc), p=2, dim=-1), w

class FollowPredictor(nn.Module):
    def __init__(self, acc_dim=NODE2VEC_DIM, hid=256):
        super().__init__()
        self.fe   = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
        self.te   = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hid*3, hid), nn.GELU(),
            nn.Linear(hid, hid//2), nn.GELU(),
            nn.Linear(hid//2, 1), nn.Sigmoid()
        )
    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)

class EndToEndFollowModel(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, account_dim=NODE2VEC_DIM, hidden_dim=256):
        super().__init__()
        self.attention_pooling = AttentionPooling(post_dim, account_dim)
        self.follow_predictor  = FollowPredictor(account_dim, hidden_dim)
    def forward(self, follower_posts, followee_posts, follower_masks=None, followee_masks=None):
        fv,_ = self.attention_pooling(follower_posts, follower_masks)
        tv,_ = self.attention_pooling(followee_posts, followee_masks)
        prob = self.follow_predictor(fv, tv)
        return prob, fv, tv

# =========================  コマンド実装  =========================
class Command(BaseCommand):
    help = "attention_pooling_follow_model.pt を使い、投稿→アカウントベクトル→フォロー推薦を一括生成します。"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="既存 UserEmbedding を削除して完全再生成")
        parser.add_argument("--top_k", type=int, default=10, help="推薦の上位件数")

    # ---------- 共通ユーティリティ ----------
    def log(self, msg, level="INFO"):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] [{level}] {msg}"
        print(line)
        self.stdout.write(line)

    def get_memory_usage(self):
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            return f"{mem:.1f}MB"
        except Exception:
            return "psutil未インストール"

    # ---------- コマンド本体 ----------
    def handle(self, *args, **opts):
        force = opts.get("force", False)
        top_k = opts.get("top_k", 10)

        self.log("="*80)
        self.log("🚀 generate_recommendations 開始")
        self.log(f"force={force} / top_k={top_k}")
        self.log("="*80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"デバイス: {device}")

        # --- モデルロード ---
        model_path = os.path.join(settings.BASE_DIR, "recommendations", "pretrained", "attention_pooling_follow_model.pt")
        model = EndToEndFollowModel().to(device).eval()
        ckpt  = torch.load(model_path, map_location="cpu")
        # prefix 換装
        ckpt = {k.replace("ap.", "attention_pooling.").replace("fp.", "follow_predictor."): v
                for k, v in ckpt.items()}
        model.load_state_dict(ckpt)

        # --- UserEmbedding リセット ---
        if force:
            cnt = UserEmbedding.objects.count()
            UserEmbedding.objects.all().delete()
            self.log(f"🗑️  旧 UserEmbedding {cnt} 件を削除（--force）")

        # --- UserEmbedding 生成 ---
        self.generate_user_embeddings(device, model)

        # --- 推薦計算 ---
        self.generate_recommendations(device, model, top_k)

        self.log("🎉 全処理完了")
        self.log("="*80)

    # ---------- 補助メソッド ----------
    def pad_posts(self, vecs, max_len=MAX_POSTS):
        n = len(vecs)
        mask = np.zeros(max_len, dtype=np.float32)
        arr  = np.zeros((max_len, OPENAI_DIM), dtype=np.float32)
        if n:
            arr[:min(n,max_len)]  = np.array(vecs[:max_len])
            mask[:min(n,max_len)] = 1
        return arr, mask

    def generate_user_embeddings(self, device, model):
        qs = User.objects.filter(is_staff=False, is_superuser=False)
        total = qs.count()
        self.log(f"UserEmbedding 再計算: 対象 {total} ユーザー")

        for idx, user in enumerate(qs, 1):
            posts = Post.objects.filter(user=user, embedding__isnull=False)
            vecs  = [np.array(p.embedding.vector, dtype=np.float32) 
                     for p in posts if len(p.embedding.vector)==OPENAI_DIM]

            if not vecs:
                continue

            arr, mask = self.pad_posts(vecs)
            with torch.no_grad():
                acc,_ = model.attention_pooling(
                    torch.tensor(arr).unsqueeze(0).to(device),
                    torch.tensor(mask).unsqueeze(0).to(device)
                )
            UserEmbedding.objects.update_or_create(
                user=user,
                defaults={"node2vec_vector": acc.squeeze(0).cpu().tolist()}
            )
            if idx<=5:
                self.log(f"  -> {user.username} 上書き完了 ({idx}/{total})")
        self.log("UserEmbedding 完了")

    def generate_recommendations(self, device, model, top_k):
        emb = UserEmbedding.objects.all()
        user_vec = {str(e.user_id): np.array(e.node2vec_vector, dtype=np.float32) for e in emb}
        users = list(user_vec.keys())

        for uid in users:
            cand = [c for c in users if c!=uid]
            results=[]
            for c in cand:
                with torch.no_grad():
                    prob = model.follow_predictor(
                        torch.tensor(user_vec[uid]).unsqueeze(0).to(device),
                        torch.tensor(user_vec[c]).unsqueeze(0).to(device)
                    ).item()
                results.append((c, prob))
            results.sort(key=lambda x: x[1], reverse=True)
            UserRecommendation.objects.filter(user_id=uid).delete()
            for rid, p in results[:top_k]:
                UserRecommendation.objects.create(
                    user_id=uid, recommended_user_id=rid,
                    score=p, follow_probability=round(p*100,1)
                )
        self.log("推薦計算 完了")