from rest_framework import serializers
from .models import Notification
from posts.serializers import UserBriefSerializer
from posts.serializers import PostSerializer

class NotificationSerializer(serializers.ModelSerializer):
    """通知シリアライザー (api_spec.yaml準拠)"""
    recipient = UserBriefSerializer(read_only=True)
    sender = UserBriefSerializer(read_only=True)
    post = PostSerializer(read_only=True)

    class Meta:
        model = Notification
        fields = [
            'id',
            'recipient',
            'sender',
            'notification_type',
            'post',
            'read',
            'created_at'
        ]
        read_only_fields = fields