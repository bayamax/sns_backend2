# /opt/sns_backend/accounts/admin.py

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Follow, Block, UserSNS

User = get_user_model()

# 既存のUser登録を解除（重複登録を避けるため）
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass

# カスタムUserAdmin定義
@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'get_sns_type', 'get_community', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'date_joined')
    search_fields = ('username', 'email')
    
    def get_sns_type(self, obj):
        """ユーザーのSNSタイプを表示"""
        try:
            return obj.sns_type.get_sns_type_display()
        except:
            return '-'
    get_sns_type.short_description = 'SNSタイプ'
    
    def get_community(self, obj):
        """ユーザーの所属コミュニティを表示"""
        try:
            membership = obj.community_membership
            status = "定住" if membership.is_settled else "放浪"
            return f"Planet {membership.community.id} ({status})"
        except:
            return '-'
    get_community.short_description = 'コミュニティ'

# UserSNS管理
@admin.register(UserSNS)
class UserSNSAdmin(admin.ModelAdmin):
    list_display = ('user', 'sns_type', 'created_at')
    list_filter = ('sns_type', 'created_at')
    search_fields = ('user__username',)

# FollowAdmin を定義
@admin.register(Follow)
class FollowAdmin(admin.ModelAdmin):
    list_display = ('follower', 'following', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('follower__username', 'following__username')

# BlockAdmin を追加
@admin.register(Block)
class BlockAdmin(admin.ModelAdmin):
    list_display = ('blocker', 'blocked', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('blocker__username', 'blocked__username')

