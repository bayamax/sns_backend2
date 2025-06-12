import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from posts.models import Post
from recommendations.models import UserEmbedding, UserRecommendation

User = get_user_model()

NODE2VEC_DIM = 128
OPENAI_DIM = 3072
MAX_POSTS = 50

class AttentionPooling(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, acc_dim=NODE2VEC_DIM, n_head=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(post_dim, n_head, batch_first=True, dropout=0.1)
        self.ln  = nn.LayerNorm(post_dim)
        self.proj = nn.Sequential(
            nn.Linear(post_dim, acc_dim*2),
            nn.GELU(),
            nn.Linear(acc_dim*2, acc_dim)
        )
        self.sc  = nn.Linear(post_dim, 1, bias=False)
    
    def forward(self, x, mask=None):
        x = F.normalize(x, p=2, dim=-1)
        att, _ = self.mha(x, x, x, key_padding_mask=(mask==0) if mask is not None else None)
        x = self.ln(x + att)
        scr = self.sc(x).squeeze(-1)
        if mask is not None:
            scr = scr.masked_fill(mask==0, -1e9)
        w = F.softmax(scr, dim=-1)
        acc = torch.sum(x * w.unsqueeze(-1), dim=1)
        return F.normalize(self.proj(acc), p=2, dim=-1), w

class FollowPredictor(nn.Module):
    def __init__(self, acc_dim=NODE2VEC_DIM, hid=256):
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

class EndToEndFollowModel(nn.Module):
    def __init__(self, post_dim=OPENAI_DIM, account_dim=NODE2VEC_DIM, hidden_dim=256):
        super().__init__()
        self.attention_pooling = AttentionPooling(post_dim, account_dim)
        self.follow_predictor = FollowPredictor(account_dim, hidden_dim)

    def forward(self, follower_posts, followee_posts, follower_masks=None, followee_masks=None):
        follower_vector, follower_attn = self.attention_pooling(follower_posts, follower_masks)
        followee_vector, followee_attn = self.attention_pooling(followee_posts, followee_masks)
        follow_prob = self.follow_predictor(follower_vector, followee_vector)
        return follow_prob, follower_vector, followee_vector, (follower_attn, followee_attn)

class Command(BaseCommand):
    help = 'attention_pooling_follow_model.ptを使って投稿ベクトル→アカウントベクトル生成→フォロー推薦まで全てワンストップで処理します。'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='既存のUserEmbeddingを全削除して再生成')
        parser.add_argument('--top_k', type=int, default=10, help='リコメンド上位件数')

    def handle(self, *args, **options):
        force = options.get('force', False)
        top_k = options.get('top_k', 10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # attention_pooling_follow_model.ptをロード
        model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'attention_pooling_follow_model.pt')
        model = EndToEndFollowModel()
        
        # state_dictのキー名を修正して読み込む
        ckpt = torch.load(model_path, map_location=device)
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('ap.'):
                new_key = k.replace('ap.', 'attention_pooling.')
                new_ckpt[new_key] = v
            elif k.startswith('fp.'):
                new_key = k.replace('fp.', 'follow_predictor.')
                new_ckpt[new_key] = v
            else:
                new_ckpt[k] = v
        
        model.load_state_dict(new_ckpt)
        model.to(device)
        model.eval()

        # 1. UserEmbeddingの再生成（投稿ベクトル→アカウントベクトル）
        if force:
            self.stdout.write(self.style.WARNING('UserEmbeddingを全削除します...'))
            UserEmbedding.objects.all().delete()
        self.generate_user_embeddings(device, model)

        # 2. フォロー推薦計算
        self.generate_recommendations(device, model, top_k)

    def pad_posts(self, posts, max_length=MAX_POSTS):
        """投稿ベクトルをパディング"""
        if len(posts) == 0:
            return np.zeros((max_length, OPENAI_DIM), dtype=np.float32), np.zeros(max_length)
        
        if len(posts) >= max_length:
            return np.array(posts[:max_length]), np.ones(max_length)
        else:
            padded = np.zeros((max_length, OPENAI_DIM), dtype=np.float32)
            mask = np.zeros(max_length)
            padded[:len(posts)] = np.array(posts)
            mask[:len(posts)] = 1
            return padded, mask

    def generate_user_embeddings(self, device, model):
        """投稿ベクトルからアカウントベクトルを生成"""
        users = User.objects.filter(is_staff=False, is_superuser=False)
        
        for user in tqdm(users, desc='アカウントベクトル生成'):
            posts = Post.objects.filter(user=user)
            if not posts.exists():
                continue
                
            # 投稿のOpenAIベクトルを取得
            post_vectors = []
            for post in posts:
                if (hasattr(post, 'embedding') and post.embedding and 
                    post.embedding.vector and isinstance(post.embedding.vector, list) and 
                    len(post.embedding.vector) == OPENAI_DIM):
                    # L2正規化を適用
                    vector = np.array(post.embedding.vector, dtype=np.float32)
                    vector_norm = np.linalg.norm(vector)
                    if vector_norm > 0:
                        normalized_vector = vector / vector_norm
                        post_vectors.append(normalized_vector)
            
            if not post_vectors:
                continue
                
            # 投稿ベクトルをパディング
            padded_posts, mask = self.pad_posts(post_vectors)
            
            # attention_poolingでアカウントベクトルを生成
            with torch.no_grad():
                posts_tensor = torch.tensor(padded_posts, dtype=torch.float32).unsqueeze(0).to(device)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                account_vector, _ = model.attention_pooling(posts_tensor, mask_tensor)
                account_vector = account_vector.squeeze(0).cpu().numpy().tolist()
            
            # UserEmbeddingを更新または新規作成
            UserEmbedding.objects.update_or_create(
                user=user,
                defaults={'node2vec_vector': account_vector}
            )

    def generate_recommendations(self, device, model, top_k):
        """アカウントベクトル間のフォロー推薦計算"""
        # 全ユーザーのアカウントベクトルを取得
        embeddings = UserEmbedding.objects.exclude(node2vec_vector__isnull=True)
        user_vectors = {str(e.user.id): np.array(e.node2vec_vector, dtype=np.float32) for e in embeddings}
        users = list(user_vectors.keys())

        for user_id in tqdm(users, desc='フォロー推薦計算'):
            # 自分以外の全ユーザーを候補に
            candidates = [uid for uid in users if uid != user_id]
            if not candidates:
                continue
                
            user_vec = user_vectors[user_id]
            results = []
            
            # バッチ処理で効率化
            batch_size = 100
            for i in range(0, len(candidates), batch_size):
                batch_candidates = candidates[i:i + batch_size]
                
                # バッチ用のテンソルを作成
                user_batch = torch.tensor([user_vec] * len(batch_candidates), dtype=torch.float32, device=device)
                candidate_batch = torch.tensor([user_vectors[c_id] for c_id in batch_candidates], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    # フォロー確率を計算
                    probs = model.follow_predictor(user_batch, candidate_batch)
                    probs_cpu = probs.cpu().numpy()
                    
                    for j, c_id in enumerate(batch_candidates):
                        results.append((c_id, probs_cpu[j]))
            
            # スコア順でソート
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:top_k]
            
            # 既存リコメンドを削除
            UserRecommendation.objects.filter(user_id=user_id).delete()
            
            # 新規保存
            for c_id, prob in top_results:
                UserRecommendation.objects.create(
                    user_id=user_id,
                    recommended_user_id=c_id,
                    score=prob,
                    follow_probability=round(min(100.0, prob * 100), 1),
                    uncertainty=0.0
                ) 