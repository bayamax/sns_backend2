# /opt/sns_backend/accounts/admin.py

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Follow # Followモデルをインポート

User = get_user_model()

# 既存のUser登録を解除（重複登録を避けるため）
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass

# カスタムUserAdmin定義
@admin.register(User)
class UserAdmin(BaseUserAdmin):
    # テスト環境で動作していた表示項目などを記述
    list_display = ('username', 'email', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'date_joined')
    search_fields = ('username', 'email')
    # fieldsets なども必要に応じて

# FollowAdmin を定義
@admin.register(Follow)
class FollowAdmin(admin.ModelAdmin):
    list_display = ('follower', 'following', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('follower__username', 'following__username')

