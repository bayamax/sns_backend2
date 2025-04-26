# recommendations/management/commands/aggregate_user_vectors.py

import os
import numpy as np
import torch
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from django.db.models import Q
from posts.models import Post
from recommendations.models import PostEmbedding, UserEmbedding
from recommendations.ml_models import AttentionAggregator

User = get_user_model()

class Command(BaseCommand):
    help = 'Aggregate post embeddings into user embeddings using attention mechanism'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reaggregation for users that already have embeddings'
        )
        parser.add_argument(
            '--user_id',
            type=int,
            help='Aggregate embeddings for a specific user (by ID)'
        )
        parser.add_argument(
            '--use_attention',
            action='store_true',
            help='Use attention mechanism for aggregation (otherwise use mean)'
        )

    def handle(self, *args, **options):
        force = options.get('force', False)
        user_id = options.get('user_id')
        use_attention = options.get('use_attention', False)
        
        # デバイスの設定
        device = self.get_device()
        self.stdout.write(f"Using device: {device}")
        
        # アテンションアグリゲータのロード
        aggregator = None
        if use_attention:
            aggregator = self.load_aggregator_model(device)
            if not aggregator:
                self.stdout.write(self.style.WARNING("Attention aggregator not available. Using mean aggregation."))
                use_attention = False
        
        # 対象ユーザーを取得
        if user_id:
            users = User.objects.filter(id=user_id)
            if not users.exists():
                self.stdout.write(self.style.ERROR(f"User with ID {user_id} not found"))
                return
        else:
            users = User.objects.all()
        
        self.stdout.write(f"Processing {users.count()} users")
        
        # 各ユーザーについて投稿の埋め込みベクトルを集約
        for user in tqdm(users, desc="Aggregating user vectors"):
            try:
                # 既存の埋め込みをスキップ（forceがFalseの場合）
                if not force and UserEmbedding.objects.filter(user=user, openai_vector__isnull=False).exists():
                    self.stdout.write(f"User {user.id}: Already has embeddings. Use --force to regenerate.")
                    continue
                
                # ユーザーの投稿を取得
                posts = Post.objects.filter(user=user)
                if not posts.exists():
                    self.stdout.write(f"User {user.id}: No posts found.")
                    continue
                
                # 投稿の埋め込みベクトルを取得
                post_vectors = []
                post_ids = []
                for post in posts:
                    try:
                        embedding = PostEmbedding.objects.get(post=post)
                        if embedding.vector:
                            post_vectors.append(np.array(embedding.vector))
                            post_ids.append(post.id)
                    except PostEmbedding.DoesNotExist:
                        pass
                
                if not post_vectors:
                    self.stdout.write(f"User {user.id}: No post embeddings found.")
                    continue
                
                # 投稿ベクトルを集約
                if use_attention and len(post_vectors) > 1:
                    # アテンションメカニズムを使用して集約
                    aggregated_vector = self.aggregate_with_attention(
                        post_vectors, aggregator, device
                    )
                    aggregation_method = "attention"
                else:
                    # 平均化による単純集約
                    aggregated_vector = np.mean(post_vectors, axis=0)
                    aggregation_method = "mean"
                
                # ユーザー埋め込みを保存
                user_embedding, created = UserEmbedding.objects.update_or_create(
                    user=user,
                    defaults={'openai_vector': aggregated_vector.tolist()}
                )
                
                self.stdout.write(
                    f"User {user.id}: Aggregated {len(post_vectors)} post vectors using {aggregation_method}"
                )
            
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing user {user.id}: {str(e)}"))
    
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
    
    def load_aggregator_model(self, device):
        """アテンションアグリゲータモデルをロード"""
        model_path = settings.MODEL_PATHS['ATTENTION_AGGREGATOR']
        
        # モデルのインスタンス化
        model = AttentionAggregator(
            input_dim=settings.RECOMMENDATION_SETTINGS['OPENAI_DIM'],
            hidden_dim=settings.RECOMMENDATION_SETTINGS['CONVERTER_HIDDEN_DIM'] // 4,
            num_heads=settings.RECOMMENDATION_SETTINGS['SET_TRANSFORMER_NUM_HEADS']
        )
        
        # 保存済みモデルがあれば読み込む
        if os.path.exists(model_path):
            self.stdout.write(f"Loading attention aggregator from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model
        else:
            self.stdout.write(self.style.WARNING(f"Attention aggregator model not found at {model_path}"))
            return None
    
    def aggregate_with_attention(self, post_vectors, aggregator, device):
        """アテンションメカニズムを使用して投稿ベクトルを集約"""
        # テンソル形式に変換
        post_tensor = torch.tensor(
            np.array(post_vectors), 
            dtype=torch.float32
        ).unsqueeze(0).to(device)  # [1, num_posts, embedding_dim]
        
        # アテンションアグリゲータを適用
        with torch.no_grad():
            aggregated = aggregator(post_tensor)
            return aggregated.squeeze(0).cpu().numpy()