# notifications/models.py
from django.db import models
from django.conf import settings

class Notification(models.Model):
    """通知モデル"""
    # 定数を明示的に定義
    LIKE = 'like'
    FOLLOW = 'follow'
    REPLY = 'reply'
    MENTION = 'mention'
    
    NOTIFICATION_TYPES = (
        (LIKE, 'いいね'),
        (FOLLOW, 'フォロー'),
        (REPLY, 'リプライ'),
        (MENTION, 'メンション'),
    )
    
    recipient = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='notifications', on_delete=models.CASCADE)
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='sent_notifications', on_delete=models.CASCADE)
    notification_type = models.CharField(max_length=10, choices=NOTIFICATION_TYPES)
    post = models.ForeignKey('posts.Post', on_delete=models.CASCADE, null=True, blank=True)
    reply_post = models.ForeignKey('posts.Post', on_delete=models.CASCADE, null=True, blank=True, related_name='reply_notifications')
    read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.sender.username} {self.get_notification_type_display()} to {self.recipient.username}"


# ------------------------------
# Push通知用デバイスモデル
# ------------------------------

class Device(models.Model):
    """APNs/Firebase などのデバイストークンを保存するモデル"""

    IOS = "ios"
    ANDROID = "android"

    PLATFORM_CHOICES = (
        (IOS, "iOS"),
        (ANDROID, "Android"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="devices",
    )
    token = models.CharField(max_length=200, unique=True)
    platform = models.CharField(max_length=20, choices=PLATFORM_CHOICES, default=IOS)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Device"
        verbose_name_plural = "Devices"

    def __str__(self):
        return f"{self.user} - {self.platform}"