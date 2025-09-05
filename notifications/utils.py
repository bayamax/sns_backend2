from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver
from django.utils import timezone
import re
from .models import Notification
from posts.models import Post, Like
from accounts.models import Follow
from django.contrib.auth import get_user_model

User = get_user_model()

# notifications/utils.py の create_notification 関数の確認・修正

def create_notification(sender, recipient, notification_type, post=None, reply_post=None):
    """通知を作成するユーティリティ関数"""
    # 自分自身への通知は作成しない
    if sender == recipient:
        return None
    
    # 同じ内容の通知が短時間内に作成されるのを防ぐ
    existing = Notification.objects.filter(
        sender=sender,
        recipient=recipient,
        notification_type=notification_type,
        post=post,
        reply_post=reply_post,
        read=False  # 未読のもののみチェック
    ).order_by('-created_at').first()
    
    # 1分以内に同じ通知が作成されていたら新規作成しない
    if existing and (timezone.now() - existing.created_at).seconds < 60:
        return existing
    
    # 新しい通知を作成
    notification = Notification.objects.create(
        sender=sender,
        recipient=recipient,
        notification_type=notification_type,
        post=post,
        reply_post=reply_post
    )

    # push 通知は post_save レシーバーで一元的に送信する
    
    return notification

# ------------------------------
# すべての通知作成後に Push 送信 (ビューから直接作られた場合も含む)
# ------------------------------


# 重複送信を防ぐためのキャッシュ
from django.core.cache import cache

from django.db.models.signals import post_save
from django.dispatch import receiver


@receiver(post_save, sender=Notification)
def send_push_after_notification_created(sender, instance, created, **kwargs):
    if not created:
        return
    # --- 重複ガード: 同一イベントは10秒以内に再送しない ---
    cache_key = f"push_lock:{instance.notification_type}:{instance.sender_id}:{instance.recipient_id}:{instance.post_id}"
    if cache.get(cache_key):
        print("[Push] duplicate suppressed", cache_key)
        return
    cache.set(cache_key, 1, timeout=10)

    try:
        from .push import send_ios_notification

        import logging
        logger = logging.getLogger(__name__)

        # Refresh recipient from DB to ensure latest devices relation
        try:
            instance.recipient.refresh_from_db(fields=None)
        except Exception:
            pass

        device_tokens = list(instance.recipient.devices.filter(platform="ios").values_list("token", flat=True))
        print("DEBUG push tokens", device_tokens)
        if not device_tokens:
            return

        title_map = {
            Notification.LIKE: "いいねされました",
            Notification.FOLLOW: "フォローされました",
            Notification.REPLY: "返信が届きました",
            Notification.MENTION: "メンションされました",
        }
        title = title_map.get(instance.notification_type, "お知らせ")
        body = instance.post.content[:50] if instance.post else "SNS"
        custom_data = {"notification_id": instance.id}
        send_ios_notification(device_tokens, title, body, custom_data)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Push notification post_save failed: {e}")


def extract_mentions(content):
    """テキストからメンション(@username)を抽出する"""
    pattern = r'@(\w+)'
    return re.findall(pattern, content)

# シグナルレシーバー
@receiver(post_save, sender=Post)
def post_created_notification(sender, instance, created, **kwargs):
    """新規投稿が作成されたときにメンションを検出して通知を作成"""
    if created:
        # 親投稿への返信の場合は通知を作成
        if instance.parent_post and instance.parent_post.user != instance.user:
            create_notification(
                sender=instance.user,
                recipient=instance.parent_post.user,
                notification_type=Notification.REPLY,
                post=instance.parent_post,
                reply_post=instance
            )
        
        # メンションを検出
        mentioned_usernames = extract_mentions(instance.content)
        for username in mentioned_usernames:
            try:
                user = User.objects.get(username=username)
                # 親投稿のユーザーには既に通知を送っている場合はスキップ
                if instance.parent_post and user == instance.parent_post.user:
                    continue
                    
                create_notification(
                    sender=instance.user,
                    recipient=user,
                    notification_type=Notification.MENTION,
                    post=instance.parent_post or instance,
                    reply_post=instance if instance.parent_post else None
                )
            except User.DoesNotExist:
                pass

@receiver(m2m_changed, sender=Post.likes.through)
def post_liked_notification(sender, instance, action, pk_set, **kwargs):
    """投稿にいいねが追加されたときに通知を作成"""
    if action == 'post_add':
        for pk in pk_set:
            try:
                user = User.objects.get(pk=pk)
                create_notification(
                    sender=user,
                    recipient=instance.user,
                    notification_type=Notification.LIKE,
                    post=instance
                )
            except User.DoesNotExist:
                pass

@receiver(post_save, sender=Follow)
def follow_created_notification(sender, instance, created, **kwargs):
    """フォローが作成されたときに通知を作成"""
    if created:
        create_notification(
            sender=instance.follower,
            recipient=instance.following,
            notification_type=Notification.FOLLOW
        )

# ------------------------------
# Like モデルが直接作成された場合（PostLikeView など）
# ------------------------------

from django.db.models.signals import post_save


# Like モデル post_save レシーバー
@receiver(post_save, sender=Like)
def like_model_created_notification(sender, instance, created, **kwargs):
    if not created:
        return
    # Like オブジェクトが保存された直後に通知作成
    if instance.post.user != instance.user:  # 自分の投稿への Like は通知しない
        create_notification(
            sender=instance.user,
            recipient=instance.post.user,
            notification_type=Notification.LIKE,
            post=instance.post,
        )