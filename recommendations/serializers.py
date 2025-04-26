# recommendations/serializers.py

from rest_framework import serializers
from .models import UserRecommendation, UserEmbedding
from accounts.serializers import UserSerializer
from django.contrib.auth import get_user_model

User = get_user_model()

class UserRecommendationSerializer(serializers.ModelSerializer):
    """ユーザー推薦シリアライザー"""
    user = UserSerializer(source='recommended_user', read_only=True)
    score = serializers.FloatField()
    followProbability = serializers.SerializerMethodField()
    uncertainty = serializers.SerializerMethodField()
    
    class Meta:
        model = UserRecommendation
        fields = ['id', 'user', 'score', 'followProbability', 'uncertainty']
    
    def get_followProbability(self, obj):
        """フォロー確率を0-1の範囲の小数で返す"""
        # データベースの値は0-100の範囲なので、100で割って0-1の範囲にする
        return obj.follow_probability / 100.0
    
    def get_uncertainty(self, obj):
        """不確実性を0-1の範囲の小数で返す"""
        # データベースの値は0-100の範囲なので、100で割って0-1の範囲にする
        return obj.uncertainty / 100.0

class UserEmbeddingSerializer(serializers.ModelSerializer):
    """ユーザー埋め込みシリアライザー"""
    class Meta:
        model = UserEmbedding
        fields = ['user', 'openai_vector', 'node2vec_vector', 'updated_at']
        read_only_fields = ['user', 'updated_at']