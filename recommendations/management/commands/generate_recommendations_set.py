import os, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import pairwise_distances_argmin
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from posts.models import Post
from recommendations.models import (
    PostEmbedding, UserEmbedding, UserRecommendation,
)

# -------------------- 設定 --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 必要に応じて環境変数で設定
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE_EMB = 32
MAX_POSTS = 100  # 1ユーザ当たり使用する最大投稿数

# set_reco_outputs ディレクトリ（codebook や学習済みモデルを置く場所）
PRETRAIN_DIR = os.path.join(settings.BASE_DIR, "recommendations", "pretrained")
CODEBOOK_PATH = os.path.join(PRETRAIN_DIR, "codebook_k512.npy")
ENCODER_CKPT = os.path.join(PRETRAIN_DIR, "checkpoint.pth")
PREDICTOR_CKPT = os.path.join(PRETRAIN_DIR, "follow_predictor.pt")

# -------------------- モデル定義 --------------------
PARAMS = dict(
    token_embed_dim=128,
    profile_dim=256,
    n_induce=16,
    n_heads=8,
    n_isab=2,
    dropout=0.0,
)

class MAB(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, Q, K, mask=None):
        q, k, v = self.q(Q), self.k(K), self.v(K)
        a, _ = self.att(q, k, v, key_padding_mask=mask)
        h = self.ln1(Q + a)
        return self.ln2(h + self.ff(h))

class ISAB(nn.Module):
    def __init__(self, d: int, heads: int, induce: int):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, induce, d))
        self.m1 = MAB(d, heads)
        self.m2 = MAB(d, heads)

    def forward(self, X, mask):
        H = self.m1(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.m2(X, H)

class PMA(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, d))
        self.m = MAB(d, heads)

    def forward(self, X, mask):
        return self.m(self.S.repeat(X.size(0), 1, 1), X, mask)

class Encoder(nn.Module):
    def __init__(self, K: int, pad_id: int):
        super().__init__()
        D = PARAMS["token_embed_dim"]
        self.emb = nn.Embedding(K + 2, D, padding_idx=pad_id)
        self.layers = nn.ModuleList(
            [ISAB(D, PARAMS["n_heads"], PARAMS["n_induce"]) for _ in range(PARAMS["n_isab"])]
        )
        self.pma = PMA(D, PARAMS["n_heads"])
        self.proj = nn.Linear(D, PARAMS["profile_dim"])

    def forward(self, x, mask):
        h = self.emb(x)
        for l in self.layers:
            h = l(h, mask)
        p = self.proj(self.pma(h, mask).squeeze(1))
        return p

class FollowPredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        d = PARAMS["profile_dim"]
        self.net = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(128, 1),
        )

    def forward(self, xy):
        return self.net(xy).squeeze(-1)

# -------------------- ユーティリティ --------------------
User = get_user_model()


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_openai(text_list):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未設定")
    import openai  # ローカル import にして依存を限定

    openai.api_key = OPENAI_API_KEY
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text_list)
    return [d["embedding"] for d in resp["data"]]

# --------------------------------------------------------

class Command(BaseCommand):
    help = "新しいセットトランスフォーマーモデルでユーザ推薦を再生成"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="既存 UserEmbedding を削除して完全再生成")
        parser.add_argument("--top_k", type=int, default=10, help="保存する推薦上位件数")

    # ---------- helpers ----------
    def log(self, msg):
        self.stdout.write(msg)

    # ---------- STEP 1: PostEmbedding 補完 ----------
    def rebuild_post_embeddings(self):
        if not OPENAI_API_KEY:
            self.log("🔍 STEP1: OPENAI_API_KEY 未設定 → 投稿埋め込み補完をスキップ")
            return

        qs = Post.objects.filter(embedding__isnull=True).values("id", "content")
        total = qs.count()
        self.log(f"🔍 STEP1: 未埋め込み投稿 = {total}")

        ids, texts, done = [], [], 0
        for rec in qs.iterator():
            ids.append(rec["id"])
            texts.append(rec["content"])
            if len(ids) == BATCH_SIZE_EMB:
                done += self._embed_batch(ids, texts)
                ids, texts = [], []
        if ids:
            done += self._embed_batch(ids, texts)
        self.log(f"✅ 投稿埋め込み補完 完了 ({done}/{total})")

    def _embed_batch(self, ids, texts):
        try:
            embeddings = embed_openai(texts)
            for pid, emb in zip(ids, embeddings):
                post = Post.objects.get(id=pid)
                PostEmbedding.objects.update_or_create(
                    post=post, defaults={"vector": emb}
                )
            return len(ids)
        except Exception as e:
            self.log(f"[ERROR] OpenAI 埋め込み失敗: {e}")
            return 0

    # ---------- STEP 2: UserEmbedding 生成 ----------
    def rebuild_user_embeddings(self, encoder, codebook, device, force):
        if force:
            cnt = UserEmbedding.objects.count()
            UserEmbedding.objects.all().delete()
            self.log(f"🗑️ 旧 UserEmbedding {cnt} 件を削除 (--force)")

        users = User.objects.filter(is_staff=False, is_superuser=False)
        total = users.count()
        self.log(f"🔍 STEP2: UserEmbedding 生成 対象 {total} 人")

        K = codebook.shape[0]
        PAD_ID = K + 1

        for idx, u in enumerate(users, 1):
            vecs = [np.array(pe.vector, np.float32)
                    for pe in PostEmbedding.objects.filter(post__user=u)
                    if pe.vector and len(pe.vector) == 3072]
            if not vecs:
                continue

            # 投稿数を制限
            vecs = vecs[:MAX_POSTS]
            code_ids = pairwise_distances_argmin(vecs, codebook)  # (L,)
            seq = torch.tensor([code_ids], dtype=torch.long, device=device)
            mask = seq == PAD_ID  # PAD 無し

            with torch.no_grad():
                prof = encoder(seq, mask).cpu().squeeze(0)
            UserEmbedding.objects.update_or_create(
                user=u, defaults={"node2vec_vector": prof.tolist()}
            )
            if idx <= 3:
                self.log(f"  -> {u.username} 更新")
        self.log("✅ UserEmbedding 生成 完了")

    # ---------- STEP 3: 推薦再計算 ----------
    def rebuild_recommendations(self, predictor, device, top_k):
        UserRecommendation.objects.all().delete()
        self.log("🗑️ 旧推薦 全削除")

        vec_map = {str(e.user_id): np.array(e.node2vec_vector, np.float32)
                   for e in UserEmbedding.objects.all() if e.node2vec_vector}
        users = list(vec_map.keys())

        for uid in users:
            cand = [c for c in users if c != uid]
            src_vec = torch.tensor(vec_map[uid], device=device).unsqueeze(0)
            concat_list = []
            cand_vecs = []
            for cid in cand:
                cand_vecs.append(vec_map[cid])
            if not cand_vecs:
                continue
            cand_tensor = torch.tensor(cand_vecs, device=device)
            src_rep = src_vec.repeat(cand_tensor.size(0), 1)
            concat = torch.cat([src_rep, cand_tensor], dim=1)
            with torch.no_grad():
                probs = torch.sigmoid(predictor(concat)).cpu().numpy()
            results = list(zip(cand, probs))
            for cid, p in sorted(results, key=lambda x: x[1], reverse=True)[:top_k]:
                UserRecommendation.objects.create(
                    user_id=uid,
                    recommended_user_id=cid,
                    score=p,
                    follow_probability=round(float(p) * 100, 1),
                )
        self.log("✅ 推薦再計算 完了")

    # ---------- メイン ----------
    def handle(self, *args, **opt):
        t0 = time.time()
        device = choose_device()
        self.log(f"🖥️ device = {device}")

        # Step 0: モデル & 資材ロード
        if not (os.path.exists(CODEBOOK_PATH) and os.path.exists(ENCODER_CKPT) and os.path.exists(PREDICTOR_CKPT)):
            self.log("[ERROR] codebook / encoder / predictor のファイルが見つかりません。PRETRAIN_DIR を確認してください。")
            return

        codebook = np.load(CODEBOOK_PATH)
        K = codebook.shape[0]
        PAD_ID = K + 1

        encoder = Encoder(K, PAD_ID)
        enc_ckpt = torch.load(ENCODER_CKPT, map_location="cpu")
        if "model" in enc_ckpt:
            enc_ckpt = enc_ckpt["model"]
        encoder.load_state_dict(enc_ckpt, strict=False)
        encoder.to(device).eval()

        predictor = FollowPredictorMLP().to(device)
        predictor.load_state_dict(torch.load(PREDICTOR_CKPT, map_location="cpu"))
        predictor.eval()

        # 1. 投稿埋め込み補完
        self.rebuild_post_embeddings()

        # 2. UserEmbedding 生成
        self.rebuild_user_embeddings(encoder, codebook, device, opt["force"])

        # 3. 推薦再計算
        self.rebuild_recommendations(predictor, device, opt["top_k"])

        self.log(f"🎉 ALL DONE in {time.time() - t0:.1f}s") 