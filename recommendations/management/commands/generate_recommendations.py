# recommendations/management/commands/generate_recommendations.py

import os
import numpy as np
import torch
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from recommendations.models import UserRecommendation, UserEmbedding
from recommendations.ml_models import ProbabilisticFollowPredictor
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ★ BOTアカウントリストを定義 (他ファイルからコピー)
BOT_ACCOUNT_USERNAMES = [
    "AIStartupDX", "ArcadeVSJP", "ArtExhibitJP", "BaseballHRJP", "BGArt_JP",
    "BirdGardenJP", "BizBookJP", "BizBookJP", "BizBookJP", "CarEnthusiastJP",
    "CasualTalkJP", "CertAIBizJP", "CoffeeCraftJP", "CompCodingJP", "ConservDXJP",
    "CosplayLiveJP", "CraftBeerJP", "DailyChatJP", "DailyVlogJP", "DartsBarJP",
    "DevPatternsJP", "EVMobilityJP", "FIRECareerJP", "FrontierTechJP", "GamerLogJP",
    "GenArtFeed", "GiveawayJP", "HistoryIFJP", "HRCareerJP", "HRWellbeingJP",
    "HYBEKpopJP", "IdolFanLogJP", "IdolFansJP", "IndieDevMaker", "IndieGameApp",
    "IndieMobileAI", "InsectBreedJP", "JLeagueHub", "JPComedyTV", "KenpoPolitics",
    "KimonoDIY", "LiveBandJP", "LocalDXJP", "LocalMuseumJP", "LocalTravelJP",
    "LoveMarryJP", "MangaPanelsJP", "MetaverseVT", "MonozukuriNet", "ObsoleteMedia",
    "OjouRoleJP", "OSSProdDev", "PortraitEvtJP", "PPTDesignLab", "PrizeWinJP",
    "ProgCareerJP", "ProVolleyJP", "RamenReportJP", "ReptileTubeJP", "RetailFastJP",
    "RhythmGameJP", "RPA_LLM", "SeiyuAudioJP", "SentoSaunaJP", "ShukatsuLab",
    "ShuroSupport", "SocBlogJP", "SoundMakersJP", "SpaceWatchJP", "StageLiveJP",
    "SubcultureJP", "TicketTradeJP", "ToyamaLocalJP", "TriviaPromoJP", "USRightWatch",
    "WatchManiaJP", "YouTubeSceneJP"
]

# SetTransformerのインポート/定義 (ここは変更なし)
try:
    from models import SetTransformer
except ImportError:
    # ... (SetTransformer 関連クラス定義は省略) ...
    import torch.nn as nn
    import torch.nn.functional as F
    class MAB(nn.Module):
        def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False): super(MAB, self).__init__(); self.dim_V = dim_V; self.num_heads = num_heads; self.fc_q = nn.Linear(dim_Q, dim_V); self.fc_k = nn.Linear(dim_K, dim_V); self.fc_v = nn.Linear(dim_K, dim_V); self.fc_o = nn.Linear(dim_V, dim_V); self.ln0 = nn.LayerNorm(dim_V) if ln else None; self.ln1 = nn.LayerNorm(dim_V) if ln else None
        def forward(self, Q, K): Q = self.fc_q(Q); K, V = self.fc_k(K), self.fc_v(K); dim_split = self.dim_V // self.num_heads; Q_ = torch.cat(Q.split(dim_split, 2), 0); K_ = torch.cat(K.split(dim_split, 2), 0); V_ = torch.cat(V.split(dim_split, 2), 0); A = torch.softmax(Q_.bmm(K_.transpose(1,2))/np.sqrt(self.dim_V), 2); O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2); O = O if self.ln0 is None else self.ln0(O); O = O + F.relu(self.fc_o(O)); O = O if self.ln1 is None else self.ln1(O); return O
    class SAB(nn.Module):
        def __init__(self, dim_in, dim_out, num_heads, ln=False): super(SAB, self).__init__(); self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        def forward(self, X): return self.mab(X, X)
    class ISAB(nn.Module):
        def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False): super(ISAB, self).__init__(); self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)); nn.init.xavier_uniform_(self.I); self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln); self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        def forward(self, X): H = self.mab0(self.I.repeat(X.size(0), 1, 1), X); return self.mab1(X, H)
    class PMA(nn.Module):
        def __init__(self, dim, num_heads, num_seeds, ln=False): super(PMA, self).__init__(); self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim)); nn.init.xavier_uniform_(self.S); self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        def forward(self, X): return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    class SetTransformer(nn.Module):
        def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=True): super(SetTransformer, self).__init__(); self.enc = nn.Sequential(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln), ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)); self.dec = nn.Sequential(PMA(dim_hidden, num_heads, 1, ln=ln), SAB(dim_hidden, dim_hidden, num_heads, ln=ln), SAB(dim_hidden, dim_hidden, num_heads, ln=ln), nn.Linear(dim_hidden, dim_output))
        def forward(self, X): return self.dec(self.enc(X))

User = get_user_model()

# 設定パラメータ (必要なものだけ残す)
NODE2VEC_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('NODE2VEC_DIM', 128)
FOLLOWEE_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FOLLOWEE_HIDDEN_DIM', 64)
MC_SAMPLES = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('MC_SAMPLES', 20)
KNN_SMOOTHING = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('KNN_SMOOTHING', 5)
SIMILARITY_WEIGHT = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SIMILARITY_WEIGHT', 0.15)
FOLLOWEE_TOP_K = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FOLLOWEE_TOP_K', 50)
FINAL_TOP_K = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FINAL_TOP_K', 10)
SET_TRANSFORMER_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_HIDDEN_DIM', 128)
SET_TRANSFORMER_NUM_HEADS = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_NUM_HEADS', 4)
SET_TRANSFORMER_NUM_INDS = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_NUM_INDS', 32)

class Command(BaseCommand):
    help = 'Generate user recommendations based on pre-calculated Node2Vec vectors, excluding bots.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user_id',
            type=int,
            help='Generate recommendations for a specific user (by ID, excluding bots)'
        )
        parser.add_argument(
            '--top_k',
            type=int,
            default=FINAL_TOP_K, # デフォルト値を定数から取得
            help='Number of recommendations to generate per user'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of recommendations even if they already exist'
        )

    def handle(self, *args, **options):
        user_id = options.get('user_id')
        top_k = options.get('top_k') # デフォルトはadd_argumentsで設定済み
        force = options.get('force')

        device = self.get_device()
        self.stdout.write(f"Using device: {device}")

        # モデルのロード (Converterは不要)
        # converter_model = self.load_converter_model(device) # 削除
        followee_model = self.load_followee_model(device)
        set_transformer_model = self.load_set_transformer_model(device)
        if not followee_model or not set_transformer_model:
             self.stdout.write(self.style.ERROR("Required models (Followee or SetTransformer) failed to load. Exiting."))
             return

        # --- ★ 処理対象ユーザーを取得 (ボットを除外) ---
        if user_id:
            users = User.objects.filter(id=user_id).exclude(username__in=BOT_ACCOUNT_USERNAMES)
            if not users.exists():
                user_check = User.objects.filter(id=user_id).first()
                if user_check and user_check.username in BOT_ACCOUNT_USERNAMES:
                     self.stdout.write(self.style.WARNING(f"User ID {user_id} ({user_check.username}) is a bot account. Skipping recommendation generation."))
                else:
                     self.stdout.write(self.style.ERROR(f"User with ID {user_id} not found or is excluded."))
                return
        else:
            users = User.objects.exclude(username__in=BOT_ACCOUNT_USERNAMES)
        # ----------------------------------------------

        user_count = users.count()
        self.stdout.write(f"Generating recommendations for {user_count} users (excluding bots)")

        # --- ★ ベクトルデータの読み込み（ボットを除外し、Node2Vecのみ使用）---
        user_vectors = self.get_user_vectors() # 引数不要に
        if not user_vectors:
            self.stdout.write(self.style.ERROR("No valid user Node2Vec vectors found. Make sure 'generate_user_node2vec' ran successfully."))
            return
        # -----------------------------------------------------------------

        node2idx, idx2node = self.build_node_mapping(user_vectors)

        # ユーザーごとにレコメンデーションを生成
        for user in tqdm(users, desc="Generating recommendations"): # tqdmにusersを渡す
            # ユーザーがベクトル辞書に存在するかチェック（ボット除外などで消えている可能性）
            if str(user.id) not in user_vectors:
                self.stdout.write(f"Skipping user {user.id}: Vector not found (likely excluded).")
                continue

            if not force and UserRecommendation.objects.filter(user=user).exists():
                # self.stdout.write(f"User {user.id}: Recommendations already exist.") # ログ抑制
                continue

            self.generate_recommendations_for_user(
                user,
                user_vectors, # ボット除外済みのベクトル辞書
                followee_model,
                set_transformer_model,
                node2idx,
                idx2node,
                device,
                top_k
            )

    def get_device(self):
        # (変更なし)
        if torch.cuda.is_available(): return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try: torch.zeros(1, device="mps"); return torch.device("mps")
            except Exception: pass
        return torch.device("cpu")

    # ★ 不要になったメソッドを削除 ★
    # def load_converter_model(self, device): ...
    # def convert_to_node2vec(self, model, openai_vector, device): ...

    def load_followee_model(self, device):
        # (中身は変更なし、エラー時の戻り値を追加)
        model_path = getattr(settings, 'MODEL_PATHS', {}).get('FOLLOWEE_MODEL', '')
        if not model_path or not os.path.exists(model_path): model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'probabilistic_followee_model.pt')
        model = ProbabilisticFollowPredictor(NODE2VEC_DIM, FOLLOWEE_HIDDEN_DIM)
        try:
             if os.path.exists(model_path):
                 self.stdout.write(f"Loading followee model from {model_path}")
                 model.load_state_dict(torch.load(model_path, map_location=device))
             else: self.stdout.write(self.style.WARNING(f"Followee model not found at {model_path}. Using untrained model."))
             model = model.to(device); model.eval()
             return model
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading followee model: {e}")); return None

    def load_set_transformer_model(self, device):
        # (中身は変更なし、エラー時の戻り値を追加)
        model_path = getattr(settings, 'MODEL_PATHS', {}).get('SET_TRANSFORMER', '')
        if not model_path or not os.path.exists(model_path): model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'set_transformer_model.pth')
        # SetTransformer の定義を再確認 (dim_output がハードコードされている点に注意)
        model = SetTransformer(dim_input=NODE2VEC_DIM, num_outputs=1, dim_output=13808, num_inds=SET_TRANSFORMER_NUM_INDS, dim_hidden=SET_TRANSFORMER_HIDDEN_DIM, num_heads=SET_TRANSFORMER_NUM_HEADS, ln=True)
        try:
             if os.path.exists(model_path):
                 self.stdout.write(f"Loading SetTransformer model from {model_path}")
                 try:
                     state_dict = torch.load(model_path, map_location=device)
                     model.load_state_dict(state_dict)
                     self.stdout.write(self.style.SUCCESS("SetTransformer loaded successfully"))
                 except Exception as e:
                     self.stdout.write(self.style.ERROR(f"Error loading SetTransformer state_dict: {str(e)}"))
                     self.stdout.write(self.style.WARNING("Using untrained SetTransformer model."))
             else: self.stdout.write(self.style.WARNING(f"SetTransformer model not found at {model_path}. Using untrained model."))
             model = model.to(device); model.eval()
             return model
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading SetTransformer model: {e}")); return None

    def build_node_mapping(self, vectors_dict):
        # (変更なし)
        node_list = sorted(vectors_dict.keys())
        node2idx = {node: i for i, node in enumerate(node_list)}
        idx2node = {i: node for node, i in node2idx.items()}
        return node2idx, idx2node

    # ★ get_user_vectors を修正: Node2Vec のみ取得、ボット除外 ★
    def get_user_vectors(self):
        """すべてのユーザー (ボット除く) の Node2Vec ベクトルを取得"""
        user_vectors = {}
        # UserEmbedding からボットを除外して取得
        embeddings = UserEmbedding.objects.exclude(user__username__in=BOT_ACCOUNT_USERNAMES)

        for embedding in embeddings:
            user_id = str(embedding.user.id)
            # node2vec_vector が存在し、正しい形式 (リストかつ次元一致) か確認
            if embedding.node2vec_vector and isinstance(embedding.node2vec_vector, list) and len(embedding.node2vec_vector) == NODE2VEC_DIM:
                try:
                    user_vectors[user_id] = np.array(embedding.node2vec_vector, dtype=np.float32)
                except ValueError:
                    self.stdout.write(self.style.WARNING(f"Could not convert Node2Vec vector for user {user_id}."))
            # openai_vector からの変換処理は削除

        found_count = len(user_vectors)
        self.stdout.write(f"Loaded {found_count} valid Node2Vec vectors (excluding bots).")
        return user_vectors
    # ---------------------------------------------------------

    # generate_recommendations_for_user: 内部の候補選択や保存時のチェックでボット除外を確認
    def generate_recommendations_for_user(self, user, user_vectors, followee_model, set_transformer_model, node2idx, idx2node, device, top_k):
        str_user_id = str(user.id)
        if str_user_id not in user_vectors:
            self.stdout.write(self.style.WARNING(f"Vector missing for user {user.id} in generate_recommendations_for_user."))
            return

        # self.stdout.write(f"Generating recommendations for user {user.id}") # ログ抑制

        # --- ★ ステップ1: フォロイー予測候補 (ここでもボットを除外すべき) ---
        # predict_followees に渡す user_vectors は既にボット除外済みだが、
        # 念のため predict_followees 側でもチェックするか、ここで再確認
        followee_candidates = self.predict_followees(
            str_user_id,
            user_vectors[str_user_id],
            user_vectors, # ボット除外済みのベクトル辞書
            followee_model,
            device,
            FOLLOWEE_TOP_K
        )

        if not followee_candidates:
            # self.stdout.write(self.style.WARNING(f"No followee candidates for user {user.id}")) # ログ抑制
            return

        followee_probs = {rec['account']: {'probability': rec['probability'], 'uncertainty': rec['uncertainty']} for rec in followee_candidates}

        # --- ★ ステップ2: SetTransformer (入力はボット除外済み) ---
        final_recommendations = self.optimize_followee_set(
            followee_candidates,
            user_vectors, # ボット除外済み
            set_transformer_model,
            node2idx,
            idx2node,
            device,
            top_k
        )

        if not final_recommendations:
            self.stdout.write(self.style.WARNING(f"Failed to optimize recommendations for user {user.id}"))
            return

        UserRecommendation.objects.filter(user=user).delete()
        saved_count = 0
        for rec in final_recommendations:
            try:
                rec_user_id = int(rec['account'])
                if rec_user_id == user.id: continue # 自分自身は除外

                try:
                    # ★ recommended_user がボットでないことを確認 ★
                    rec_user = User.objects.exclude(username__in=BOT_ACCOUNT_USERNAMES).get(id=rec_user_id)
                except User.DoesNotExist:
                    # ボットだったか、存在しないユーザー
                    # self.stdout.write(f"Skipping bot or non-existent user: {rec_user_id}") # ログ抑制
                    continue
                # -------------------------------------------

                follow_prob = followee_probs.get(rec['account'], {}).get('probability', 0.0); follow_prob = max(0.0, min(1.0, follow_prob))
                uncertainty = followee_probs.get(rec['account'], {}).get('uncertainty', 0.0); uncertainty = max(0.0, min(1.0, uncertainty))
                raw_score = rec.get('transformer_score', 0.0); normalized_score = max(min(raw_score, 1000), -1000)

                UserRecommendation.objects.create(
                    user=user,
                    recommended_user=rec_user,
                    score=normalized_score,
                    follow_probability=round(min(100.0, follow_prob * 100), 1),
                    uncertainty=round(min(100.0, uncertainty * 100), 1)
                )
                saved_count += 1
            except (ValueError, KeyError) as e:
                self.stdout.write(f"Error processing recommendation: {e}")

        # self.stdout.write(f"Saved {saved_count} recommendations for user {user.id}") # ログ抑制

    # predict_followees: 入力 user_vectors がボット除外済みであることを前提とする
    def predict_followees(self, user_id, user_vector, user_vectors, model, device, top_k):
        # 候補アカウント（user_vectors は既にボット除外済みのはず）
        candidate_vectors = {acc_id: vector for acc_id, vector in user_vectors.items() if acc_id != user_id}
        if not candidate_vectors: return []

        similarity_matrix = self.compute_similarities(user_vector, user_vectors, user_id)
        mc_predictions = self.monte_carlo_predictions(model, user_vector, candidate_vectors, device)
        smoothed_distribution = self.smoothed_preference_distribution(mc_predictions, similarity_matrix, user_vectors, candidate_vectors)

        max_prob = max(smoothed_distribution.values()) if smoothed_distribution else 1.0
        normalized_distribution = {acc_id: min(1.0, prob / max_prob) for acc_id, prob in smoothed_distribution.items()} if max_prob > 0 else smoothed_distribution

        sorted_predictions = sorted(normalized_distribution.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for acc_id, prob in sorted_predictions:
            uncertainty = mc_predictions.get(acc_id, {}).get('std', 0.0)
            uncertainty = min(1.0, uncertainty / max_prob) if max_prob > 0 else uncertainty
            result.append({'account': acc_id, 'probability': float(prob), 'uncertainty': float(uncertainty)})
        return result

    # optimize_followee_set: 入力 user_vectors がボット除外済みであることを前提とする
    def optimize_followee_set(self, followee_candidates, user_vectors, set_transformer_model, node2idx, idx2node, device, top_k):
        # self.stdout.write("Optimizing followee set...") # ログ抑制
        all_vectors = []; all_user_ids = []
        for acc_id, vector in user_vectors.items(): all_vectors.append(vector); all_user_ids.append(acc_id)
        if not all_vectors: return []

        input_vectors = np.array(all_vectors)
        # ★ SetTransformerの入力次元がNODE2VEC_DIMであることを確認
        if input_vectors.shape[1] != NODE2VEC_DIM:
             self.stdout.write(self.style.ERROR(f"Input vector dimension for SetTransformer is incorrect ({input_vectors.shape[1]} vs {NODE2VEC_DIM})"))
             return [] # エラーケース
        input_tensor = torch.tensor(input_vectors, dtype=torch.float, device=device).unsqueeze(0)

        try:
            with torch.no_grad():
                scores = set_transformer_model(input_tensor)
                # self.stdout.write(f"SetTransformer output shape: {scores.shape}") # ログ抑制
                scores = scores.view(-1).cpu().numpy()
                candidate_scores = {}
                candidate_ids = {c['account'] for c in followee_candidates} # 高速化のためセットに
                for i, acc_id in enumerate(all_user_ids):
                    if acc_id in candidate_ids and i < len(scores):
                        candidate_scores[acc_id] = float(scores[i])

                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                result = [{'account': acc_id, 'transformer_score': score} for acc_id, score in sorted_candidates]

                if not result:
                    self.stdout.write(self.style.WARNING("SetTransformer returned no results, using fallback."))
                    return followee_candidates[:top_k]
                return result
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during SetTransformer inference: {str(e)}"))
            import traceback; self.stdout.write(self.style.ERROR(traceback.format_exc()))
            return followee_candidates[:top_k] # フォールバック

    # compute_similarities: 入力 all_vectors がボット除外済みであることを前提とする
    def compute_similarities(self, user_vector, all_vectors, user_id):
        similarities = {}
        user_vector = user_vector.reshape(1, -1)
        for acc_id, vector in all_vectors.items():
            if acc_id != user_id:
                vector = vector.reshape(1, -1)
                try: # ベクトル計算エラーをキャッチ
                     sim = cosine_similarity(user_vector, vector)[0][0]
                     similarities[acc_id] = sim
                except ValueError:
                     self.stdout.write(self.style.WARNING(f"Could not compute similarity between {user_id} and {acc_id}"))
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:KNN_SMOOTHING]
        return dict(top_similar)

    # monte_carlo_predictions: 入力 candidate_vectors がボット除外済みであることを前提とする
    def monte_carlo_predictions(self, model, user_vector, candidate_vectors, device):
        model.train()
        user_tensor = torch.tensor(user_vector, dtype=torch.float32).unsqueeze(0).to(device)
        results = {}
        # tqdm の desc をより具体的に
        for acc_id, vector in tqdm(candidate_vectors.items(), desc="MC Predictions", leave=False):
            candidate_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
            probs = []
            try: # モデル推論エラーをキャッチ
                 with torch.no_grad():
                     for _ in range(MC_SAMPLES):
                         prob = model(user_tensor, candidate_tensor)
                         probs.append(prob.item())
                 mean_prob = np.mean(probs); std_prob = np.std(probs)
                 results[acc_id] = {'mean': mean_prob, 'std': std_prob}
            except Exception as e:
                 self.stdout.write(self.style.ERROR(f"Error during MC prediction for {acc_id}: {e}"))
        return results

    # smoothed_preference_distribution: 入力 all_vectors, candidate_vectors がボット除外済みであることを前提とする
    def smoothed_preference_distribution(self, mc_predictions, similarity_matrix, all_vectors, candidate_vectors):
        distribution = {acc_id: predictions.get('mean', 0.0) for acc_id, predictions in mc_predictions.items()}
        for similar_acc, sim_score in similarity_matrix.items():
            if similar_acc in distribution:
                for acc_id in candidate_vectors.keys():
                    if similar_acc in all_vectors and acc_id in all_vectors:
                        try: # ベクトル計算エラーをキャッチ
                             acc_sim = cosine_similarity(all_vectors[similar_acc].reshape(1, -1), all_vectors[acc_id].reshape(1, -1))[0][0]
                             influence = distribution[similar_acc] * sim_score * acc_sim * SIMILARITY_WEIGHT
                             # 0除算を避ける
                             denominator = (1 + SIMILARITY_WEIGHT)
                             if denominator > 1e-9:
                                 distribution[acc_id] = (distribution.get(acc_id, 0.0) + influence) / denominator
                        except ValueError:
                             self.stdout.write(self.style.WARNING(f"Could not compute similarity in smoothing for {acc_id}"))
        return distribution