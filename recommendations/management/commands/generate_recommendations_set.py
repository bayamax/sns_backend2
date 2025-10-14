import os, time, numpy as np, torch, torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from posts.models import Post
from recommendations.models import (
    PostEmbedding, UserEmbedding, UserRecommendation,
)
from django.db.models import Q
from accounts.models import UserSNS, Follow

# -------------------- 設定 --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 必要に応じて環境変数で設定
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE_EMB = 32
MAX_POSTS = 100  # 1ユーザ当たり使用する最大投稿数

# set_reco_outputs ディレクトリ（codebook や学習済みモデルを置く場所）
PRETRAIN_DIR = os.path.join(settings.BASE_DIR, "recommendations", "pretrained")

# --- 最新 codebook_k*.npy を自動検出 -----------------------------
def find_latest_codebook(dir_path: str) -> str:
    """codebook_kXXXX.npy の中で最大 XXXX を返す"""
    try:
        files = [f for f in os.listdir(dir_path) if f.startswith("codebook_k") and f.endswith(".npy")]
    except FileNotFoundError:
        files = []
    if not files:
        raise FileNotFoundError(f"codebook_k*.npy が {dir_path} に見つかりません")
    latest = max(files, key=lambda s: int(s.split("_k")[1].split(".")[0]))
    return os.path.join(dir_path, latest)

def find_codebook_by_k(dir_path: str, k: int) -> Optional[str]:
    path = os.path.join(dir_path, f"codebook_k{k}.npy")
    return path if os.path.exists(path) else None

def resolve_predictor_ckpt(dir_path: str) -> Optional[str]:
    search_dirs = [
        dir_path,
        os.path.join(settings.BASE_DIR, "follow_predictor_outputs"),
        os.path.join(settings.BASE_DIR, "recommendations", "follow_predictor_outputs"),
    ]
    for d in search_dirs:
        # 正しいモデルが follow_mlp.pt のため、こちらを優先
        for name in ("follow_mlp.pt", "follow_predictor.pt"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    return None

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
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d)
        )

    def forward(self, Q, K, mask=None):
        q, k, v = self.q(Q), self.k(K), self.v(K)
        a, _ = self.att(q, k, v, key_padding_mask=mask)
        h = self.ln1(Q + a)
        return self.ln2(h + self.ff(h))

class ISAB(nn.Module):
    def __init__(self, d: int, heads: int, induce: int):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, induce, d))
        self.m1 = MAB(d, heads, PARAMS["dropout"])
        self.m2 = MAB(d, heads, PARAMS["dropout"])

    def forward(self, X, mask):
        H = self.m1(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.m2(X, H)

class PMA(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, d))
        self.m = MAB(d, heads, PARAMS["dropout"])

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
        # 学習時と同じヘッド/ドロップアウトを持たせて厳密ロードに一致させる
        self.cls = nn.Linear(D, K)
        self.drop = nn.Dropout(PARAMS["dropout"])

    def forward(self, x, mask):
        h = self.emb(x)
        for l in self.layers:
            h = l(h, mask)
        _ = self.cls(self.drop(h))  # 学習時互換のために前向きに通すが出力は使用しない
        p = self.proj(self.drop(self.pma(h, mask).squeeze(1)))
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
        # 本アプリにのみ帰属するアカウントを対象にするためのフィルタ
        # accounts.models.UserSNS で定義されている sns_type と合わせる
        parser.add_argument(
            "--sns_type",
            type=str,
            default="threadplanet",
            help="対象とするSNSタイプ (例: 'map' or 'threadplanet')",
        )

    # ---------- helpers ----------
    def log(self, msg):
        self.stdout.write(msg)

    # ---------- STEP 1: PostEmbedding 補完 ----------
    def rebuild_post_embeddings(self):
        if not OPENAI_API_KEY:
            self.log("🔍 STEP1: OPENAI_API_KEY 未設定 → 投稿埋め込み補完をスキップ")
            return

        # 本アプリ所属ユーザの投稿のみに限定
        qs = (
            Post.objects.filter(embedding__isnull=True)
            .filter(self.sns_q)
            .values("id", "content")
        )
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

        # 対象SNSタイプのユーザに限定
        users = (
            User.objects.filter(is_staff=False, is_superuser=False)
            .filter(self.sns_user_q)
        )
        total = users.count()
        self.log(f"🔍 STEP2: UserEmbedding 生成 対象 {total} 人")

        K = codebook.shape[0]
        PAD_ID = K + 1

        for idx, u in enumerate(users, 1):
            vecs = [
                np.array(pe.vector, np.float32)
                for pe in PostEmbedding.objects.filter(post__user=u).order_by('-post__created_at')
                if pe.vector and len(pe.vector) == 3072
            ]
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
        # 既存推薦を対象SNSタイプのユーザ分のみ削除
        UserRecommendation.objects.filter(self.sns_q).delete()
        self.log("🗑️ 旧推薦 (指定SNSタイプ) を削除")

        vec_map = {
            str(e.user_id): np.array(e.node2vec_vector, np.float32)
            for e in UserEmbedding.objects.filter(self.sns_q)
            if e.node2vec_vector
        }
        users = list(vec_map.keys())

        # 事前にフォロー関係を収集して、生成段階でフォロー相手を除外
        # 対象ユーザー群に限定したフォロー関係のマップを作る
        user_ids_int = [int(uid) for uid in users]
        following_map = {}
        if user_ids_int:
            following_map = {str(uid): set() for uid in user_ids_int}
            for rec in Follow.objects.filter(follower_id__in=user_ids_int, following_id__in=user_ids_int).values("follower_id", "following_id"):
                follower = str(rec["follower_id"])
                following = str(rec["following_id"])
                following_map.setdefault(follower, set()).add(following)

        for uid in users:
            # 自分自身と、既にフォローしているユーザーを候補から除外
            followed = following_map.get(uid, set())
            cand = [c for c in users if c != uid and c not in followed]
            src_vec = F.normalize(torch.tensor(vec_map[uid], device=device), dim=0).unsqueeze(0)
            cand_vecs = []
            for cid in cand:
                cand_vecs.append(F.normalize(torch.tensor(vec_map[cid], device=device), dim=0))
            if not cand_vecs:
                continue
            cand_tensor = torch.stack(cand_vecs)
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

        # 対象SNSタイプを保持
        self.target_sns = opt.get("sns_type", "threadplanet")

        # UserSNS レコードが存在するかで null 取り扱いを切替
        labeled_exists = UserSNS.objects.exists()
        self.sns_q = Q(user__sns_type__sns_type=self.target_sns) if labeled_exists else (
            Q(user__sns_type__sns_type=self.target_sns) | Q(user__sns_type__isnull=True)
        )
        # User モデルに適用するフィルタは "user__" を外した形で保持
        self.sns_user_q = Q(sns_type__sns_type=self.target_sns) if labeled_exists else (
            Q(sns_type__sns_type=self.target_sns) | Q(sns_type__isnull=True)
        )

        # Step 0: モデル & 資材ロード
        if not os.path.exists(ENCODER_CKPT):
            self.log(f"[ERROR] encoder checkpoint が見つかりません: {ENCODER_CKPT}")
            return

        # 0-1) エンコーダ ckpt から K を推定
        raw_ckpt = torch.load(ENCODER_CKPT, map_location="cpu")
        enc_state = raw_ckpt.get("model", raw_ckpt)
        # DataParallel 由来の 'module.' 接頭辞を除去
        if isinstance(enc_state, dict) and any(k.startswith("module.") for k in enc_state.keys()):
            enc_state = {k.replace("module.", "", 1): v for k, v in enc_state.items()}
        K_from_ckpt = None
        for k, v in enc_state.items():
            if k.endswith("emb.weight") and hasattr(v, "shape"):
                # emb の語彙サイズは K+2（PAD含む）
                K_from_ckpt = int(v.shape[0]) - 2
                break

        # 0-2) codebook を K に合わせて解決
        codebook_path = None
        if K_from_ckpt is not None:
            codebook_path = find_codebook_by_k(PRETRAIN_DIR, K_from_ckpt)
            if not codebook_path:
                # 一致するものが無ければ最新を使うが、後で不一致ならエラーにする
                try:
                    codebook_path = find_latest_codebook(PRETRAIN_DIR)
                except Exception as e:
                    self.log(f"[ERROR] codebook が見つかりません: {e}")
                    return
        else:
            # K が取れなかった場合は従来どおり最新を使う
            try:
                codebook_path = find_latest_codebook(PRETRAIN_DIR)
            except Exception as e:
                self.log(f"[ERROR] codebook が見つかりません: {e}")
                return

        if not codebook_path or not os.path.exists(codebook_path):
            self.log(f"[ERROR] codebook パスを解決できませんでした: {codebook_path}")
            return

        codebook = np.load(codebook_path)
        K = int(codebook.shape[0])
        if K_from_ckpt is not None and K_from_ckpt != K:
            self.log(
                f"[ERROR] codebook の次元 K={K} と checkpoint の K={K_from_ckpt} が不一致です。"
                f" 対応する codebook_k{K_from_ckpt}.npy を {PRETRAIN_DIR} に配置してください。"
            )
            return

        PAD_ID = K + 1

        encoder = Encoder(K, PAD_ID)
        try:
            encoder.load_state_dict(enc_state, strict=True)
        except Exception as e:
            self.log(f"[ERROR] Encoder の重み読み込みに失敗しました (strict=True): {e}")
            return
        encoder.to(device).eval()

        # 0-3) 予測器 ckpt 解決（follow_predictor.pt / follow_mlp.pt どちらでも）
        predictor_ckpt_path = resolve_predictor_ckpt(PRETRAIN_DIR)
        if not predictor_ckpt_path:
            self.log(f"[ERROR] predictor のチェックポイントが見つかりません ({PRETRAIN_DIR})")
            return
        raw_pred = torch.load(predictor_ckpt_path, map_location="cpu")
        if isinstance(raw_pred, dict) and ("model" in raw_pred or "state_dict" in raw_pred):
            pred_state = raw_pred.get("model", raw_pred.get("state_dict", raw_pred))
        else:
            pred_state = raw_pred
        # DataParallel 由来の 'module.' 接頭辞を除去
        if isinstance(pred_state, dict) and any(k.startswith("module.") for k in pred_state.keys()):
            pred_state = {k.replace("module.", "", 1): v for k, v in pred_state.items()}

        predictor = FollowPredictorMLP().to(device)
        try:
            predictor.load_state_dict(pred_state, strict=True)
        except Exception as e:
            self.log(f"[ERROR] Predictor の重み読み込みに失敗しました (strict=True): {e}")
            return
        predictor.eval()

        self.log(f"📦 使用 codebook: {os.path.basename(codebook_path)} (K={K})")
        self.log(f"📦 使用 encoder ckpt: {os.path.basename(ENCODER_CKPT)} (K={K_from_ckpt or K})")
        self.log(f"📦 使用 predictor ckpt: {os.path.basename(predictor_ckpt_path)}")

        # 1. 投稿埋め込み補完
        self.rebuild_post_embeddings()

        # 2. UserEmbedding 生成
        self.rebuild_user_embeddings(encoder, codebook, device, opt["force"])

        # 3. 推薦再計算
        self.rebuild_recommendations(predictor, device, opt["top_k"])

        self.log(f"🎉 ALL DONE in {time.time() - t0:.1f}s")
