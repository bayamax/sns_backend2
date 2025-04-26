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