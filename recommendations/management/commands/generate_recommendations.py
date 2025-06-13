# -*- coding: utf-8 -*-
"""
update_recommendations.py

1. OpenAI で投稿埋め込みを補完（未生成投稿のみ）
2. AttentionPooling で UserEmbedding を再生成
3. FollowPredictor で推薦を再計算（旧推薦を全削除 → 上位 k 件保存）
"""

import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from posts.models import Post                      # 投稿本体
from recommendations.models import (               # 埋め込み & 推薦
    PostEmbedding, UserEmbedding, UserRecommendation
)

# ───── 編集ポイント ──────────────────────────────
OPENAI_API_KEY = ""               # ここに sk-xxxx を貼る（空なら環境変数を使用）
EMBED_MODEL    = "text-embedding-3-large"
BATCH_SIZE_EMB = 32               # OpenAI 呼び出しのバッチサイズ
# ────────────────────────────────────────────────

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    print("[WARNING] OPENAI_API_KEY が設定されていないため、投稿埋め込み補完はスキップされます。")

User = get_user_model()
NODE2VEC_DIM, OPENAI_DIM, MAX_POSTS = 128, 3072, 50

# -------------------------- モデル定義 --------------------------
class AttentionPooling(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, acc_dim=NODE2VEC_DIM, n_head=8):
        super().__init__()
        self.mha  = nn.MultiheadAttention(post_dim, n_head, batch_first=True, dropout=0.1)
        self.ln   = nn.LayerNorm(post_dim)
        self.proj = nn.Sequential(nn.Linear(post_dim, acc_dim*2), nn.GELU(),
                                  nn.Linear(acc_dim*2, acc_dim))
        self.sc   = nn.Linear(post_dim, 1, bias=False)

    def forward(self, x, mask=None):
        x  = nn.functional.normalize(x, p=2, dim=-1)
        att,_ = self.mha(x, x, x, key_padding_mask=(mask==0) if mask is not None else None)
        x  = self.ln(x + att)
        scr = self.sc(x).squeeze(-1)
        if mask is not None:
            scr = scr.masked_fill(mask == 0, -1e9)
        w   = nn.functional.softmax(scr, dim=-1)
        acc = torch.sum(x * w.unsqueeze(-1), dim=1)
        return nn.functional.normalize(self.proj(acc), p=2, dim=-1)

class FollowPredictor(nn.Module):
    def __init__(self, dim=NODE2VEC_DIM, hid=256):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.te = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hid*3, hid), nn.GELU(),
            nn.Linear(hid, hid//2), nn.GELU(),
            nn.Linear(hid//2, 1), nn.Sigmoid()
        )
    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)

class EndToEndFollowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_pooling = AttentionPooling()
        self.follow_predictor  = FollowPredictor()

# -------------------------- コマンド --------------------------
class Command(BaseCommand):
    help = "OpenAI埋め込み→UserEmbedding→推薦を一括更新"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true",
                            help="既存 UserEmbedding を削除して完全再生成")
        parser.add_argument("--top_k", type=int, default=10,
                            help="保存する推薦上位件数")

    def log(self, msg): self.stdout.write(msg)

    # ---------- STEP 1: 投稿埋め込み補完 ----------
    def rebuild_post_embeddings(self):
        if not OPENAI_API_KEY:
            self.log("🔍 STEP1: キーなし → 補完をスキップ")
            return

        qs = Post.objects.filter(embedding__isnull=True).values("id", "content")
        total = qs.count()
        self.log(f"🔍 STEP1: 未埋め込み投稿 = {total}")

        ids, texts, done = [], [], 0
        for rec in qs.iterator():
            ids.append(rec["id"]); texts.append(rec["content"])
            if len(ids) == BATCH_SIZE_EMB:
                done += self._embed_batch(ids, texts)
                ids, texts = [], []
        if ids:
            done += self._embed_batch(ids, texts)
        self.log(f"✅ 投稿埋め込み補完 完了 ({done}/{total})")

    def _embed_batch(self, ids, texts):
        try:
            resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
            for pid, item in zip(ids, resp["data"]):
                post = Post.objects.get(id=pid)
                PostEmbedding.objects.update_or_create(
                    post=post,
                    defaults={"vector": item["embedding"]}
                )
            return len(ids)
        except Exception as e:
            self.log(f"[ERROR] OpenAI 埋め込み失敗: {e}")
            return 0

    # ---------- STEP 2: UserEmbedding 生成 ----------
    def rebuild_user_embeddings(self, model, device, force):
        if force:
            cnt = UserEmbedding.objects.count()
            UserEmbedding.objects.all().delete()
            self.log(f"🗑️ 旧 UserEmbedding {cnt} 件を削除 (--force)")

        users = User.objects.filter(is_staff=False, is_superuser=False)
        total = users.count()
        self.log(f"🔍 STEP2: UserEmbedding 生成 対象 {total} 人")

        for idx, u in enumerate(users, 1):
            vecs = [np.array(pe.vector, np.float32)
                    for pe in PostEmbedding.objects.filter(post__user=u)
                    if pe.vector and len(pe.vector)==OPENAI_DIM]
            if not vecs: continue
            arr, mask = self._pad(vecs)
            with torch.no_grad():
                acc = model.attention_pooling(
                    torch.tensor(arr).unsqueeze(0).to(device),
                    torch.tensor(mask).unsqueeze(0).to(device)).cpu().squeeze(0)
            UserEmbedding.objects.update_or_create(
                user=u, defaults={"node2vec_vector": acc.tolist()})
            if idx <= 3:
                self.log(f"  -> {u.username} 更新")
        self.log("✅ UserEmbedding 生成 完了")

    def _pad(self, vecs):
        n = min(len(vecs), MAX_POSTS)
        arr  = np.zeros((MAX_POSTS, OPENAI_DIM), np.float32)
        mask = np.zeros(MAX_POSTS, np.float32)
        arr[:n]  = vecs[:n]
        mask[:n] = 1
        return arr, mask

    # ---------- STEP 3: 推薦再計算 ----------
    def rebuild_recommendations(self, model, device, top_k):
        UserRecommendation.objects.all().delete()
        self.log("🗑️ 旧推薦 全削除")

        vec_map = {str(e.user_id): np.array(e.node2vec_vector, np.float32)
                   for e in UserEmbedding.objects.all()}
        users = list(vec_map.keys())

        for uid in users:
            cand = [c for c in users if c != uid]
            results = []
            for cid in cand:
                with torch.no_grad():
                    p = model.follow_predictor(
                            torch.tensor(vec_map[uid]).unsqueeze(0).to(device),
                            torch.tensor(vec_map[cid]).unsqueeze(0).to(device)).item()
                results.append((cid, p))
            for cid, p in sorted(results, key=lambda x: x[1], reverse=True)[:top_k]:
                UserRecommendation.objects.create(
                    user_id=uid, recommended_user_id=cid,
                    score=p, follow_probability=round(p*100, 1))
        self.log("✅ 推薦再計算 完了")

    # ---------- エントリーポイント ----------
    def handle(self, *args, **opt):
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"🖥️ device = {device}")

        # 1. 投稿埋め込み補完
        self.rebuild_post_embeddings()

        # 2. モデルロード（学習済みあれば）
        model = EndToEndFollowModel()
        ckpt_path = os.path.join(settings.BASE_DIR,
                                 "recommendations", "pretrained",
                                 "attention_pooling_follow_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt = {k.replace("ap.", "attention_pooling.").replace("fp.", "follow_predictor."): v
                    for k, v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
        model.to(device).eval()

        # 3. UserEmbedding → 推薦
        self.rebuild_user_embeddings(model, device, opt["force"])
        self.rebuild_recommendations(model, device, opt["top_k"])

        self.log(f"🎉 ALL DONE in {time.time()-t0:.1f}s")