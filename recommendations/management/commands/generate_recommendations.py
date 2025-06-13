# -*- coding: utf-8 -*-
"""
update_recommendations.py

OpenAI で投稿埋め込み → UserEmbedding 再生成 → 推薦再計算 を一括実行
"""

import os, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from posts.models import Post
from recommendations.models import UserEmbedding, UserRecommendation

# ───── editable ────────────────────────────────────────────────
OPENAI_API_KEY = ""                 # ← ここにキーを直書きするか環境変数を利用
EMBED_MODEL    = "text-embedding-3-large"
BATCH_SIZE_EMB = 32                 # OpenAI へのバッチサイズ
# ─────────────────────────────────────────────────────────────

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    print("[WARNING] OPENAI_API_KEY が設定されていません。投稿埋め込み補完はスキップします。")

User = get_user_model()
NODE2VEC_DIM, OPENAI_DIM, MAX_POSTS = 128, 3072, 50

# ----------------- モデル定義 -----------------
class AttentionPooling(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, acc_dim=NODE2VEC_DIM, n_head=8):
        super().__init__()
        self.mha  = nn.MultiheadAttention(post_dim, n_head, batch_first=True, dropout=0.1)
        self.ln   = nn.LayerNorm(post_dim)
        self.proj = nn.Sequential(nn.Linear(post_dim, acc_dim*2), nn.GELU(),
                                  nn.Linear(acc_dim*2, acc_dim))
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
        return F.normalize(self.proj(acc), p=2, dim=-1)

class FollowPredictor(nn.Module):
    def __init__(self, dim=NODE2VEC_DIM, hid=256):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.te = nn.Sequential(nn.Linear(dim, hid), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hid*3, hid), nn.GELU(),
            nn.Linear(hid, hid//2), nn.GELU(),
            nn.Linear(hid//2, 1), nn.Sigmoid())

    def forward(self, f, t):
        f, t = self.fe(f), self.te(t)
        return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)

class EndToEndFollowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_pooling = AttentionPooling()
        self.follow_predictor  = FollowPredictor()

# ----------------- コマンド -----------------
class Command(BaseCommand):
    help = "OpenAI埋め込み補完→UserEmbedding再生成→推薦再計算を一括実行"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="旧UserEmbeddingも削除して完全再生成")
        parser.add_argument("--top_k", type=int, default=10, help="保存する推薦上位数")

    # ------------- 共通 -------------
    def log(self, msg): self.stdout.write(msg)
    def pad_posts(self, vecs):
        arr = np.zeros((MAX_POSTS, OPENAI_DIM), np.float32)
        mask= np.zeros(MAX_POSTS, np.float32)
        n   = min(len(vecs), MAX_POSTS)
        if n: arr[:n] = vecs[:n]; mask[:n] = 1
        return arr, mask

    # ------------- STEP 1: 投稿埋め込み補完 -------------
    def rebuild_post_embeddings(self):
        if not OPENAI_API_KEY:
            self.log("🔍 STEP1: キーなし→投稿埋め込み補完をスキップ")
            return

        qs = Post.objects.filter(embedding__isnull=True).values("id", "content")
        total = qs.count()
        self.log(f"🔍 STEP1: 埋め込み未生成投稿={total}")

        batch_ids, batch_text = [], []
        done = 0
        for rec in qs.iterator():
            batch_ids.append(rec["id"])
            batch_text.append(rec["content"])
            if len(batch_ids)==BATCH_SIZE_EMB:
                done += self._embed_batch(batch_ids, batch_text)
                batch_ids, batch_text = [], []
        if batch_ids:
            done += self._embed_batch(batch_ids, batch_text)
        self.log(f"✅ 投稿埋め込み補完 完了 ({done}/{total})")

    def _embed_batch(self, ids, texts):
        try:
            resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
            for pid, item in zip(ids, resp["data"]):
                Post.objects.filter(id=pid).update(embedding={"vector": item["embedding"]})
            return len(ids)
        except Exception as e:
            self.log(f"[ERROR] OpenAI 埋め込み失敗: {e}")
            return 0

    # ------------- STEP 2: UserEmbedding 再生成 -------------
    def rebuild_user_embeddings(self, model, device, force):
        if force:
            cnt = UserEmbedding.objects.count()
            UserEmbedding.objects.all().delete()
            self.log(f"🗑️ 旧UserEmbedding {cnt}件削除 (--force)")

        users = User.objects.filter(is_staff=False, is_superuser=False)
        total = users.count()
        self.log(f"🔍 STEP2: UserEmbedding 生成 対象 {total} 人")

        for i, u in enumerate(users, 1):
            posts = Post.objects.filter(user=u, embedding__isnull=False)
            vecs  = [np.array(p.embedding.vector, np.float32) for p in posts if len(p.embedding.vector)==OPENAI_DIM]
            if not vecs: continue
            arr, mask = self.pad_posts(vecs)
            with torch.no_grad():
                acc = model.attention_pooling(
                    torch.tensor(arr).unsqueeze(0).to(device),
                    torch.tensor(mask).unsqueeze(0).to(device)).cpu().squeeze(0)
            UserEmbedding.objects.update_or_create(
                user=u, defaults={"node2vec_vector": acc.tolist()})
            if i<=3: self.log(f"  -> {u.username} 更新")
        self.log("✅ UserEmbedding 生成完了")

    # ------------- STEP 3: 推薦再計算 -------------
    def rebuild_recommendations(self, model, device, top_k):
        all_vec = {str(e.user_id): np.array(e.node2vec_vector, np.float32)
                   for e in UserEmbedding.objects.all()}
        users = list(all_vec.keys())
        UserRecommendation.objects.all().delete()
        self.log(f"🗑️ 旧推薦 全削除")

        for uid in users:
            cand = [c for c in users if c!=uid]
            results=[]
            for cid in cand:
                with torch.no_grad():
                    p = model.follow_predictor(
                            torch.tensor(all_vec[uid]).unsqueeze(0).to(device),
                            torch.tensor(all_vec[cid]).unsqueeze(0).to(device)).item()
                results.append((cid, p))
            for cid, p in sorted(results, key=lambda x:x[1], reverse=True)[:top_k]:
                UserRecommendation.objects.create(
                    user_id=uid, recommended_user_id=cid, score=p,
                    follow_probability=round(p*100,1))
        self.log("✅ 推薦再計算 完了")

    # ------------- handle() -------------
    def handle(self, *args, **opt):
        start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"🖥️ device={device}")
        self.rebuild_post_embeddings()

        model = EndToEndFollowModel()
        # 事前学習重みの読み込み （存在しない場合はランダム初期化）
        ckpt_path = os.path.join(settings.BASE_DIR, "recommendations", "pretrained", "attention_pooling_follow_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt = {k.replace("ap.", "attention_pooling.").replace("fp.","follow_predictor."):v for k,v in ckpt.items()}
            model.load_state_dict(ckpt, strict=False)
        model.to(device).eval()

        self.rebuild_user_embeddings(model, device, opt["force"])
        self.rebuild_recommendations(model, device, opt["top_k"])
        self.log(f"🎉 ALL DONE in {time.time()-start:.1f}s")