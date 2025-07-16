# posts/models.py

from django.db import models
from django.conf import settings

class Post(models.Model):
    """投稿モデル"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='posts')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    likes = models.ManyToManyField(settings.AUTH_USER_MODEL, through='Like')
    parent_post = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='post_replies')
    replies_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        if self.parent_post:
            return f"{self.user.username}'s reply to post {self.parent_post.id}"
        return f"{self.user.username}'s post ({self.id})"
    
    def likes_count(self):
        return self.likes.count()
    
    @property
    def is_reply(self):
        return self.parent_post is not None

class Like(models.Model):
    """いいねモデル"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='likes')
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'post')
    
    def __str__(self):
        return f"{self.user.username} likes post {self.post.id}"

class Report(models.Model):
    """投稿の報告"""
    REPORT_REASONS = ( # 必要に応じて理由を追加・変更
        ('spam', 'スパム'),
        ('inappropriate', '不適切なコンテンツ'),
        ('harassment', '嫌がらせ'),
        ('other', 'その他'),
    )
    STATUS_CHOICES = (
        ('pending', '未対応'),
        ('resolved', '対応済み'),
        ('ignored', '無視'),
    )

    reporter = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='reports_made', on_delete=models.CASCADE)
    reported_post = models.ForeignKey('Post', related_name='reports_received', on_delete=models.CASCADE)
    reason = models.CharField(max_length=20, choices=REPORT_REASONS, blank=True, null=True)
    detail = models.TextField(blank=True, null=True) # 報告詳細（任意）
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    class Meta:
        unique_together = ('reporter', 'reported_post') # 同じユーザーが同じ投稿を複数回報告するのを防ぐ
        ordering = ['-timestamp']

    def __str__(self):
        return f"Report by {self.reporter.username} on post {self.reported_post.id}"


class PostLocation(models.Model):
    """投稿に紐づく位置情報を保持するモデル"""
    post = models.OneToOneField(Post, on_delete=models.CASCADE, related_name='post_location')
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    place_name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Location({self.latitude}, {self.longitude}) for Post {self.post_id}"