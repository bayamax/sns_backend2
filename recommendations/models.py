# recommendations/models.py

from django.db import models
from django.conf import settings

class UserRecommendation(models.Model):
    """ユーザー推薦モデル"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='recommended_to'
    )
    recommended_user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='recommended_as'
    )
    score = models.FloatField(default=0.0)  # 総合推薦スコア
    follow_probability = models.FloatField(default=0.0)  # フォロー確率
    uncertainty = models.FloatField(default=0.0)  # 不確実性
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('user', 'recommended_user')
        ordering = ['-score']
    
    def __str__(self):
        return f"Recommendation of {self.recommended_user.username} to {self.user.username}"

class PostEmbedding(models.Model):
    """投稿の埋め込みベクトルモデル"""
    post = models.OneToOneField('posts.Post', on_delete=models.CASCADE, related_name='embedding')
    vector = models.JSONField(null=True, blank=True)  # OpenAI埋め込みベクトル
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Embedding for post {self.post.id}"

class UserEmbedding(models.Model):
    """ユーザーの埋め込みベクトルモデル"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='embedding')
    openai_vector = models.JSONField(null=True, blank=True)  # OpenAI埋め込みベクトル（集約済み）
    node2vec_vector = models.JSONField(null=True, blank=True)  # Node2Vecベクトル
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Embeddings for user {self.user.username}"

class Community(models.Model):
    """コミュニティモデル"""
    name = models.CharField(max_length=100, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name or f"Community {self.id}"

class CommunityMembership(models.Model):
    """ユーザーのコミュニティ所属を管理するモデル"""
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='community_membership'
    )
    community = models.ForeignKey(
        Community,
        on_delete=models.CASCADE,
        related_name='members'
    )
    assigned_at = models.DateTimeField(auto_now=True)
    is_settled = models.BooleanField(default=False)  # 定住フラグ
    settled_at = models.DateTimeField(null=True, blank=True)  # 定住日時
    
    def __str__(self):
        status = "定住" if self.is_settled else "放浪"
        return f"{self.user.username} in Community {self.community.id} ({status})"