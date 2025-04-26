# posts/serializers.py

from rest_framework import serializers
from .models import Post, Like
from django.contrib.auth import get_user_model

User = get_user_model()

class UserBriefSerializer(serializers.ModelSerializer):
    """簡易的なユーザー情報のシリアライザー（投稿用）"""
    profileImageUrl = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'profileImageUrl', 'bio']
    
    def get_profileImageUrl(self, obj):
        return obj.profile_image_url

class PostSerializer(serializers.ModelSerializer):
    """投稿シリアライザー（YAML仕様準拠：snake_caseフィールド）"""
    user_id = serializers.SerializerMethodField()
    user = UserBriefSerializer(read_only=True)
    content = serializers.CharField()
    created_at = serializers.DateTimeField(read_only=True)
    updated_at = serializers.DateTimeField(read_only=True)
    likes_count = serializers.SerializerMethodField()
    replies_count = serializers.SerializerMethodField()
    is_liked = serializers.SerializerMethodField()
    child_replies = serializers.SerializerMethodField()
    is_reply = serializers.SerializerMethodField()
    parent_post = serializers.PrimaryKeyRelatedField(read_only=True)
    is_from_followed_user = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            'id',
            'user_id',
            'user',
            'content',
            'created_at',
            'updated_at',
            'likes_count',
            'replies_count',
            'is_liked',
            'child_replies',
            'is_reply',
            'parent_post',
            'is_from_followed_user',
        ]
        read_only_fields = fields

    def get_user_id(self, obj):
        return obj.user.id

    def get_likes_count(self, obj):
        return obj.likes.count()

    def get_replies_count(self, obj):
        return obj.replies_count

    def get_is_liked(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.likes.filter(id=request.user.id).exists()
        return False

    def get_is_reply(self, obj):
        return obj.is_reply

    def get_child_replies(self, obj):
        if 'no_children' in self.context:
            return []
        child_replies = obj.post_replies.all().order_by('-created_at')[:5]
        child_context = self.context.copy() if self.context else {}
        child_context['no_children'] = True
        return PostSerializer(child_replies, many=True, context=child_context).data

    def get_is_from_followed_user(self, obj):
        return getattr(obj, 'is_from_followed_user', False)

class PostCreateSerializer(serializers.ModelSerializer):
    """投稿作成シリアライザー"""
    parent_post = serializers.PrimaryKeyRelatedField(queryset=Post.objects.all(), required=False, allow_null=True)
    
    class Meta:
        model = Post
        fields = ['content', 'parent_post']
    
    def create(self, validated_data):
        user = self.context['request'].user
        parent_post = validated_data.get('parent_post')
        
        # 新しい投稿を作成
        post = Post.objects.create(
            user=user,
            content=validated_data.get('content'),
            parent_post=parent_post
        )
        
        # 親投稿がある場合、replies_countを増やす
        if parent_post:
            parent_post.replies_count += 1
            parent_post.save(update_fields=['replies_count'])
        
        return post

class LikeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Like
        fields = ['id', 'user', 'post', 'created_at']