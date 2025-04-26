# recommendations/management/commands/generate_recommendations.py

import os
import numpy as np
import torch
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from recommendations.models import UserRecommendation, UserEmbedding
from recommendations.ml_models import EmbeddingConverter, ProbabilisticFollowPredictor
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# SetTransformerのインポートを追加
try:
    from models import SetTransformer
except ImportError:
    # SetTransformerクラスを直接実装
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MAB(nn.Module):
        def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
            super(MAB, self).__init__()
            self.dim_V = dim_V
            self.num_heads = num_heads
            self.fc_q = nn.Linear(dim_Q, dim_V)
            self.fc_k = nn.Linear(dim_K, dim_V)
            self.fc_v = nn.Linear(dim_K, dim_V)
            if ln:
                self.ln0 = nn.LayerNorm(dim_V)
                self.ln1 = nn.LayerNorm(dim_V)
            self.fc_o = nn.Linear(dim_V, dim_V)

        def forward(self, Q, K):
            Q = self.fc_q(Q)
            K, V = self.fc_k(K), self.fc_v(K)

            dim_split = self.dim_V // self.num_heads
            Q_ = torch.cat(Q.split(dim_split, 2), 0)
            K_ = torch.cat(K.split(dim_split, 2), 0)
            V_ = torch.cat(V.split(dim_split, 2), 0)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/np.sqrt(self.dim_V), 2)
            O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
            O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
            O = O + F.relu(self.fc_o(O))
            O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
            return O

    class SAB(nn.Module):
        def __init__(self, dim_in, dim_out, num_heads, ln=False):
            super(SAB, self).__init__()
            self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

        def forward(self, X):
            return self.mab(X, X)

    class ISAB(nn.Module):
        def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
            super(ISAB, self).__init__()
            self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
            nn.init.xavier_uniform_(self.I)
            self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
            self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

        def forward(self, X):
            H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
            return self.mab1(X, H)

    class PMA(nn.Module):
        def __init__(self, dim, num_heads, num_seeds, ln=False):
            super(PMA, self).__init__()
            self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
            nn.init.xavier_uniform_(self.S)
            self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

        def forward(self, X):
            return self.mab(self.S.repeat(X.size(0), 1, 1), X)

    class SetTransformer(nn.Module):
        def __init__(self, dim_input, num_outputs, dim_output,
                     num_inds=32, dim_hidden=128, num_heads=4, ln=True):
            super(SetTransformer, self).__init__()
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            )
            self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, 1, ln=ln),  # num_seedsを1に固定
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output)
            )

        def forward(self, X):
            return self.dec(self.enc(X))

User = get_user_model()

# 設定パラメータ（settings.pyから取得するか、デフォルト値を使用）
OPENAI_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('OPENAI_DIM', 3072)
NODE2VEC_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('NODE2VEC_DIM', 128)
CONVERTER_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('CONVERTER_HIDDEN_DIM', 1024)
FOLLOWEE_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FOLLOWEE_HIDDEN_DIM', 64)
MC_SAMPLES = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('MC_SAMPLES', 20)
KNN_SMOOTHING = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('KNN_SMOOTHING', 5)
SIMILARITY_WEIGHT = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SIMILARITY_WEIGHT', 0.15)
FOLLOWEE_TOP_K = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FOLLOWEE_TOP_K', 50)
FINAL_TOP_K = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FINAL_TOP_K', 10)

# SetTransformer関連のパラメータ
SET_TRANSFORMER_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_HIDDEN_DIM', 128)
SET_TRANSFORMER_NUM_HEADS = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_NUM_HEADS', 4)
SET_TRANSFORMER_NUM_INDS = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('SET_TRANSFORMER_NUM_INDS', 32)

class Command(BaseCommand):
    help = 'Generate user recommendations based on embedding vectors'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user_id',
            type=int,
            help='Generate recommendations for a specific user (by ID)'
        )
        parser.add_argument(
            '--top_k',
            type=int,
            default=10,
            help='Number of recommendations to generate per user'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of recommendations even if they already exist'
        )

    def handle(self, *args, **options):
        user_id = options.get('user_id')
        top_k = options.get('top_k', FINAL_TOP_K)
        force = options.get('force')
        
        # デバイスの設定
        device = self.get_device()
        self.stdout.write(f"Using device: {device}")
        
        # モデルのロード
        converter_model = self.load_converter_model(device)
        followee_model = self.load_followee_model(device)
        set_transformer_model = self.load_set_transformer_model(device)
        
        # 処理対象のユーザーを取得
        if user_id:
            users = User.objects.filter(id=user_id)
            if not users.exists():
                self.stdout.write(self.style.ERROR(f"User with ID {user_id} not found"))
                return
        else:
            users = User.objects.all()
        
        self.stdout.write(f"Generating recommendations for {users.count()} users")
        
        # ベクトルデータの読み込み（一度だけ）
        user_vectors = self.get_user_vectors(converter_model, device)
        if not user_vectors:
            self.stdout.write(self.style.ERROR("No user vectors found. Make sure to run aggregate_user_vectors first."))
            return
            
        # ノードIDマッピングの構築
        node2idx, idx2node = self.build_node_mapping(user_vectors)
        
        # ユーザーごとにレコメンデーションを生成
        for user in users:
            if not force and UserRecommendation.objects.filter(user=user).exists():
                self.stdout.write(f"Recommendations already exist for user {user.id}. Use --force to regenerate.")
                continue
            
            self.generate_recommendations_for_user(
                user, 
                user_vectors, 
                followee_model, 
                set_transformer_model,
                node2idx,
                idx2node,
                device, 
                top_k
            )
    
    def get_device(self):
        """利用可能なデバイスを取得"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device="mps")
                return torch.device("mps")
            except:
                pass
        return torch.device("cpu")
    
    def load_converter_model(self, device):
        """OpenAI→Node2Vec変換モデルをロード"""
        model_path = getattr(settings, 'MODEL_PATHS', {}).get('EMBEDDING_CONVERTER', '')
        
        if not model_path or not os.path.exists(model_path):
            model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'openai_to_node2vec_model.pt')
        
        model = EmbeddingConverter(OPENAI_DIM, NODE2VEC_DIM, CONVERTER_HIDDEN_DIM)
        
        if os.path.exists(model_path):
            self.stdout.write(f"Loading converter model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            self.stdout.write(self.style.WARNING(f"Converter model not found at {model_path}. Using untrained model."))
        
        model = model.to(device)
        model.eval()
        return model
    
    def load_followee_model(self, device):
        """フォロイー予測モデルをロード"""
        model_path = getattr(settings, 'MODEL_PATHS', {}).get('FOLLOWEE_MODEL', '')
        
        if not model_path or not os.path.exists(model_path):
            model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'probabilistic_followee_model.pt')
        
        model = ProbabilisticFollowPredictor(NODE2VEC_DIM, FOLLOWEE_HIDDEN_DIM)
        
        if os.path.exists(model_path):
            self.stdout.write(f"Loading followee model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            self.stdout.write(self.style.WARNING(f"Followee model not found at {model_path}. Using untrained model."))
        
        model = model.to(device)
        model.eval()
        return model
    
    def load_set_transformer_model(self, device):
        """SetTransformerモデルをロード"""
        model_path = getattr(settings, 'MODEL_PATHS', {}).get('SET_TRANSFORMER', '')
        
        if not model_path or not os.path.exists(model_path):
            model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'set_transformer_model.pth')
        
        # 実際のモデル構造に合わせる
        model = SetTransformer(
            dim_input=NODE2VEC_DIM,
            num_outputs=1,  # PMAのシード数を1に固定
            dim_output=13808,  # 出力次元を13808に設定
            num_inds=SET_TRANSFORMER_NUM_INDS,
            dim_hidden=SET_TRANSFORMER_HIDDEN_DIM,
            num_heads=SET_TRANSFORMER_NUM_HEADS,
            ln=True
        )
        
        if os.path.exists(model_path):
            self.stdout.write(f"Loading SetTransformer model from {model_path}")
            try:
                # 保存済みモデルのパラメータを読み込み
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                self.stdout.write(self.style.SUCCESS("モデルを正常にロードしました"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error loading SetTransformer model: {str(e)}"))
                self.stdout.write(self.style.WARNING("モデルのロードに失敗しました。未学習のモデルを使用します。"))
        else:
            self.stdout.write(self.style.WARNING(f"SetTransformer model not found at {model_path}. Using untrained model."))
        
        model = model.to(device)
        model.eval()
        return model
    
    def build_node_mapping(self, vectors_dict):
        """アカウントID → インデックス, インデックス → アカウントID のマッピングを作成"""
        node_list = sorted(vectors_dict.keys())
        node2idx = {node: i for i, node in enumerate(node_list)}
        idx2node = {i: node for node, i in node2idx.items()}
        return node2idx, idx2node
    
    def convert_to_node2vec(self, model, openai_vector, device):
        """OpenAIベクトルをNode2Vecに変換"""
        with torch.no_grad():
            tensor = torch.tensor(openai_vector, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(tensor)
            return output.squeeze(0).cpu().numpy()
    
    def get_user_vectors(self, converter_model, device):
        """すべてのユーザーのベクトルを取得"""
        user_vectors = {}
        
        # 埋め込みベクトルを持つすべてのユーザーを取得
        embeddings = UserEmbedding.objects.all()
        
        for embedding in embeddings:
            user_id = str(embedding.user.id)
            
            # Node2Vecベクトルがあればそれを使用
            if embedding.node2vec_vector:
                user_vectors[user_id] = np.array(embedding.node2vec_vector)
            # なければOpenAIベクトルから変換
            elif embedding.openai_vector:
                openai_vector = np.array(embedding.openai_vector)
                node2vec_vector = self.convert_to_node2vec(converter_model, openai_vector, device)
                user_vectors[user_id] = node2vec_vector
                
                # 変換したNode2Vecベクトルを保存（再利用のため）
                embedding.node2vec_vector = node2vec_vector.tolist()
                embedding.save()
        
        self.stdout.write(f"Loaded vectors for {len(user_vectors)} users")
        return user_vectors
        
    def generate_recommendations_for_user(self, user, user_vectors, followee_model, set_transformer_model, node2idx, idx2node, device, top_k):
        """特定ユーザーのレコメンデーションを生成"""
        str_user_id = str(user.id)
        
        if str_user_id not in user_vectors:
            self.stdout.write(self.style.WARNING(f"No vector found for user {user.id}"))
            return
            
        self.stdout.write(f"Generating recommendations for user {user.id}")
        
        # ステップ1: フォロイー予測候補の生成 (SetTransformerの入力用)
        followee_candidates = self.predict_followees(
            str_user_id, 
            user_vectors[str_user_id], 
            user_vectors, 
            followee_model, 
            device, 
            FOLLOWEE_TOP_K
        )
        
        if not followee_candidates:
            self.stdout.write(self.style.WARNING(f"No followee candidates generated for user {user.id}"))
            return
        
        # フォローする確率とその不確実性を保存
        followee_probs = {
            rec['account']: {
                'probability': rec['probability'],
                'uncertainty': rec['uncertainty']
            }
            for rec in followee_candidates
        }
            
        # ステップ2: SetTransformerを使用して最適なセットを生成 (最終結果)
        final_recommendations = self.optimize_followee_set(
            followee_candidates,
            user_vectors,
            set_transformer_model,
            node2idx,
            idx2node,
            device,
            top_k
        )
        
        if not final_recommendations:
            self.stdout.write(self.style.WARNING(f"Failed to optimize recommendations for user {user.id}"))
            return
        
        # 既存のレコメンデーションを削除
        UserRecommendation.objects.filter(user=user).delete()
        
        # 新しいレコメンデーションを保存
        saved_count = 0
        for rec in final_recommendations:
            try:
                # ユーザーIDを取得
                rec_user_id = int(rec['account'])
                
                # ユーザーが存在するか確認
                if rec_user_id == user.id:  # 自分自身は除外
                    continue
                    
                try:
                    rec_user = User.objects.get(id=rec_user_id)
                except User.DoesNotExist:
                    continue
                
                # フォロー確率と不確実性を取得（0〜1の範囲に制限）
                follow_prob = followee_probs.get(rec['account'], {}).get('probability', 0.0)
                follow_prob = max(0.0, min(1.0, follow_prob))  # 0〜1の範囲に制限
                
                uncertainty = followee_probs.get(rec['account'], {}).get('uncertainty', 0.0)
                uncertainty = max(0.0, min(1.0, uncertainty))  # 0〜1の範囲に制限
                
                # スコアを正規化（-1000〜1000の範囲に収める）
                raw_score = rec.get('transformer_score', 0.0)
                normalized_score = max(min(raw_score, 1000), -1000)
                
                # レコメンデーションを保存（0〜100の範囲にスケーリング）
                # 値は小数第1位までの整数値として保存（10.0%→10）
                UserRecommendation.objects.create(
                    user=user,
                    recommended_user=rec_user,
                    score=normalized_score,
                    follow_probability=round(min(100.0, follow_prob * 100), 1),  # 小数第1位までの値
                    uncertainty=round(min(100.0, uncertainty * 100), 1)  # 小数第1位までの値
                )
                saved_count += 1
            except (ValueError, KeyError) as e:
                self.stdout.write(f"Error processing recommendation: {e}")
        
        self.stdout.write(f"Saved {saved_count} recommendations for user {user.id}")
    
    def predict_followees(self, user_id, user_vector, user_vectors, model, device, top_k):
        """フォロイー候補を予測"""
        # 候補アカウント（自分以外）
        candidate_vectors = {
            acc_id: vector 
            for acc_id, vector in user_vectors.items() 
            if acc_id != user_id
        }
        
        if not candidate_vectors:
            self.stdout.write(self.style.WARNING("No candidate users found"))
            return []
        
        # 類似アカウントの計算
        similarity_matrix = self.compute_similarities(user_vector, user_vectors, user_id)
        
        # MCドロップアウトによる予測
        mc_predictions = self.monte_carlo_predictions(
            model, user_vector, candidate_vectors, device
        )
        
        # 類似アカウントを考慮した分布
        smoothed_distribution = self.smoothed_preference_distribution(
            mc_predictions, similarity_matrix, user_vectors, candidate_vectors
        )
        
        # 確率値を0〜1の範囲に正規化
        max_prob = max(smoothed_distribution.values()) if smoothed_distribution else 1.0
        if max_prob > 0:
            # 最大値で割って正規化
            normalized_distribution = {
                acc_id: min(1.0, prob / max_prob) 
                for acc_id, prob in smoothed_distribution.items()
            }
        else:
            normalized_distribution = smoothed_distribution
        
        # ソートして上位を取得
        sorted_predictions = sorted(
            normalized_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # 結果をフォーマット
        result = []
        for acc_id, prob in sorted_predictions:
            uncertainty = mc_predictions.get(acc_id, {}).get('std', 0.0)
            # 不確実性も正規化
            if max_prob > 0:
                uncertainty = min(1.0, uncertainty / max_prob)
            
            result.append({
                'account': acc_id,
                'probability': float(prob),
                'uncertainty': float(uncertainty)
            })
        
        return result
    
    def optimize_followee_set(self, followee_candidates, user_vectors, set_transformer_model, node2idx, idx2node, device, top_k):
        """SetTransformerを使用してフォロイーセットを最適化"""
        self.stdout.write("Optimizing followee set using SetTransformer...")
        
        # 全ユーザーの埋め込みベクトルを取得
        all_vectors = []
        all_user_ids = []
        for acc_id, vector in user_vectors.items():
            all_vectors.append(vector)
            all_user_ids.append(acc_id)

        if not all_vectors:
            self.stdout.write(self.style.ERROR("No valid input vectors found"))
            return []

        # Tensor に変換 (batch=1, set_size=..., embed_dim=...)
        input_vectors = np.array(all_vectors)
        input_tensor = torch.tensor(input_vectors, dtype=torch.float, device=device).unsqueeze(0)

        # 推論
        try:
            with torch.no_grad():
                scores = set_transformer_model(input_tensor)  # shape: (batch=1, 1, output_dim=13808)
                self.stdout.write(f"Output tensor shape: {scores.shape}")
                
                # 形状を修正：(1, 1, 13808) -> (13808,)
                scores = scores.view(-1).cpu().numpy()
                
                # 候補アカウントのスコアを抽出
                candidate_scores = {}
                for acc_id in followee_candidates:
                    if acc_id['account'] in all_user_ids:
                        idx = all_user_ids.index(acc_id['account'])
                        if idx < len(scores):  # インデックスの範囲チェック
                            candidate_scores[acc_id['account']] = float(scores[idx])
                
                # スコアの高い順に top_k を取得
                sorted_candidates = sorted(
                    candidate_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                
                # 結果をフォーマット
                result = []
                for acc_id, score in sorted_candidates:
                    result.append({
                        'account': acc_id,
                        'transformer_score': score
                    })
                
                if not result:
                    self.stdout.write(self.style.WARNING("SetTransformer returned no valid results, using fallback"))
                    return followee_candidates[:top_k]
                    
                return result
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during SetTransformer inference: {str(e)}"))
            import traceback
            self.stdout.write(self.style.ERROR(traceback.format_exc()))
            # フォールバック: 単純なフォロイー候補上位を返す
            return followee_candidates[:top_k]
    
    def compute_similarities(self, user_vector, all_vectors, user_id):
        """ユーザー間の類似度を計算"""
        similarities = {}
        user_vector = user_vector.reshape(1, -1)
        
        for acc_id, vector in all_vectors.items():
            if acc_id != user_id:
                vector = vector.reshape(1, -1)
                sim = cosine_similarity(user_vector, vector)[0][0]
                similarities[acc_id] = sim
        
        # 上位K個を取得
        top_similar = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:KNN_SMOOTHING]
        
        return dict(top_similar)
    
    def monte_carlo_predictions(self, model, user_vector, candidate_vectors, device):
        """モンテカルロドロップアウトによる確率的予測"""
        model.train()  # ドロップアウトを有効化
        user_tensor = torch.tensor(user_vector, dtype=torch.float32).unsqueeze(0).to(device)
        
        results = {}
        for acc_id, vector in tqdm(candidate_vectors.items(), desc="Processing predictions"):
            candidate_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
            probs = []
            
            # 複数回サンプリング
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    prob = model(user_tensor, candidate_tensor)
                    probs.append(prob.item())
            
            # 統計量の計算
            mean_prob = np.mean(probs)
            std_prob = np.std(probs)
            
            results[acc_id] = {
                'mean': mean_prob,
                'std': std_prob
            }
        
        return results
    
    def smoothed_preference_distribution(self, mc_predictions, similarity_matrix, all_vectors, candidate_vectors):
        """類似アカウントの影響を考慮した滑らかな確率分布"""
        # 基本の確率分布
        distribution = {
            acc_id: predictions.get('mean', 0.0) 
            for acc_id, predictions in mc_predictions.items()
        }
        
        # 類似アカウントの影響を加味
        for similar_acc, sim_score in similarity_matrix.items():
            if similar_acc in distribution:
                for acc_id in candidate_vectors.keys():
                    if similar_acc in all_vectors and acc_id in all_vectors:
                        # 類似アカウントと候補アカウントの類似度
                        acc_sim = cosine_similarity(
                            all_vectors[similar_acc].reshape(1, -1),
                            all_vectors[acc_id].reshape(1, -1)
                        )[0][0]
                        
                        # 影響の計算
                        influence = distribution[similar_acc] * sim_score * acc_sim * SIMILARITY_WEIGHT
                        distribution[acc_id] = (distribution[acc_id] + influence) / (1 + SIMILARITY_WEIGHT)
        
        return distribution