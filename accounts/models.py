# accounts/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils.translation import gettext_lazy as _
from django.conf import settings

class UserManager(BaseUserManager):
    """カスタムユーザーマネージャー"""
    def create_user(self, username, email=None, password=None, **extra_fields):
        if not username:
            raise ValueError(_('ユーザー名は必須です'))
        
        email = self.normalize_email(email) if email else None
        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('スーパーユーザーはis_staff=Trueでなければなりません'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('スーパーユーザーはis_superuser=Trueでなければなりません'))
            
        return self.create_user(username, email, password, **extra_fields)

class User(AbstractUser):
    """カスタムユーザーモデル"""
    email = models.EmailField(_('メールアドレス'), blank=True, null=True)
    username = models.CharField(_('ユーザー名'), max_length=150, unique=True)
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True)
    bio = models.TextField(_('自己紹介'), blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Apple サインイン専用アカウントかどうか
    is_apple_only = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)
    
    objects = UserManager()
    
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = []
    
    def __str__(self):
        return self.username
    
    @property
    def followers_count(self):
        return self.followers.count()
    
    @property
    def following_count(self):
        return self.following.count()
    
    @property
    def profile_image_url(self):
        if self.profile_image:
            return self.profile_image.url
        return None

class Follow(models.Model):
    """フォロー関係を表すモデル"""
    follower = models.ForeignKey(User, related_name='following', on_delete=models.CASCADE)
    following = models.ForeignKey(User, related_name='followers', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('follower', 'following')
        
    def __str__(self):
        return f"{self.follower.username} follows {self.following.username}"

class Block(models.Model):
    """ユーザー間のブロック関係"""
    blocker = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='blocking', on_delete=models.CASCADE)
    blocked = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='blocked_by', on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('blocker', 'blocked') # 同じ組み合わせでの重複ブロックを防ぐ
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.blocker.username} blocked {self.blocked.username}"
