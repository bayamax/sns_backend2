# recommendations/ml_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingConverter(nn.Module):
    """OpenAI埋め込みをNode2Vecベクトルに変換するモデル"""
    def __init__(self, openai_dim, node2vec_dim, hidden_dim):
        super(EmbeddingConverter, self).__init__()
        
        # 3層のMLPと正規化層、活性化関数を使用
        self.model = nn.Sequential(
            nn.Linear(openai_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, node2vec_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class ProbabilisticFollowPredictor(nn.Module):
    """確率的フォロー関係予測モデル"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(ProbabilisticFollowPredictor, self).__init__()
        
        # フォロワーのエンコーダー
        self.follower_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # フォロイー候補のエンコーダー
        self.followee_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 中間層
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def encode_follower(self, follower_vector):
        """フォロワーエンコーディング"""
        return self.follower_encoder(follower_vector)
    
    def encode_followee(self, candidate_vector):
        """フォロイー候補エンコーディング"""
        return self.followee_encoder(candidate_vector)
    
    def forward(self, follower_vector, candidate_vector):
        """フォロー確率予測"""
        follower_features = self.encode_follower(follower_vector)
        candidate_features = self.encode_followee(candidate_vector)
        
        # 特徴の結合
        combined_features = torch.cat([follower_features, candidate_features], dim=1)
        
        # 中間表現
        hidden = self.hidden_layer(combined_features)
        
        # フォロー確率
        follow_prob = self.output_layer(hidden).squeeze()
        
        return follow_prob

class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション実装"""
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output_linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Q, K, Vの計算
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # スケーリングドットプロダクトアテンション
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # アテンション適用
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        output = self.output_linear(context)
        
        return output

class AttentionAggregator(nn.Module):
    """投稿ベクトルを集約するアテンション集約器"""
    def __init__(self, input_dim, hidden_dim=256, num_heads=4):
        super(AttentionAggregator, self).__init__()
        
        self.attention = MultiHeadAttention(input_dim, num_heads)
        
        # コンテキスト行列（クエリとして機能）
        self.context = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 最終出力層
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, post_vectors):
        """
        post_vectors: [batch_size, num_posts, embedding_dim]
        """
        batch_size, num_posts, _ = post_vectors.size()
        
        # コンテキストベクトルをバッチサイズに拡張
        context = self.context.expand(batch_size, 1, -1)
        
        # 投稿とコンテキストを結合
        combined = torch.cat([context, post_vectors], dim=1)
        
        # セルフアテンション
        attended = self.attention(combined)
        
        # コンテキストベクトルだけ取得（集約結果）
        aggregated = attended[:, 0]
        
        # 最終変換
        output = self.output_layer(aggregated)
        
        return output

class SetTransformer(nn.Module):
    """最適な推薦セットを生成するためのSetTransformer"""
    def __init__(self, input_dim, hidden_dim, num_heads, num_inds, num_outputs=10):
        super(SetTransformer, self).__init__()
        
        # インダクションポイントを使ったアテンション
        self.inductor = nn.Parameter(torch.randn(1, num_inds, input_dim))
        
        # エンコーダ層
        self.encoder = nn.Sequential(
            MultiHeadAttention(input_dim, num_heads),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # デコーダ層（出力シード）
        self.decoder_seed = nn.Parameter(torch.randn(1, num_outputs, input_dim))
        
        # デコーダ層
        self.decoder = nn.Sequential(
            MultiHeadAttention(input_dim, num_heads),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # スコア層
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, followee_vectors, candidate_vectors):
        """
        followee_vectors: 仮想フォロイーセット [batch_size, num_followees, embedding_dim]
        candidate_vectors: 候補アカウント [batch_size, num_candidates, embedding_dim]
        """
        batch_size, num_followees, _ = followee_vectors.size()
        _, num_candidates, _ = candidate_vectors.size()
        
        # インダクタをバッチサイズに拡張
        inds = self.inductor.expand(batch_size, -1, -1)
        
        # フォロイーセットをエンコード（インダクションポイント経由）
        # インダクションポイントとフォロイーのクロスアテンションを計算
        combined = torch.cat([inds, followee_vectors], dim=1)
        encoded = self.encoder(combined)
        inds_encoded = encoded[:, :inds.size(1)]
        
        # デコーダシードをバッチサイズに拡張
        seed = self.decoder_seed.expand(batch_size, -1, -1)
        
        # デコード処理（シードとエンコード済みインダクタのクロスアテンション）
        decoded = self.decoder(torch.cat([seed, inds_encoded], dim=1))
        query_vectors = decoded[:, :seed.size(1)]
        
        # 各候補アカウントに対するスコア計算
        scores = []
        for i in range(num_candidates):
            # 候補ベクトルをクエリベクトルと結合
            cand = candidate_vectors[:, i:i+1].expand(-1, query_vectors.size(1), -1)
            combined = torch.cat([query_vectors, cand], dim=-1)
            # スコア計算
            score = self.score(combined).squeeze(-1)
            scores.append(score)
        
        # スコアを積み重ねて返す [batch_size, num_outputs, num_candidates]
        stacked_scores = torch.stack(scores, dim=-1)
        
        return stacked_scores