import os
import random
import subprocess
import sys
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
from .models import Post

# 確率は環境変数で調整可能 (デフォルト 5%)
TRIGGER_PROB = float(os.getenv("RECO_TRIGGER_PROB", "0.2"))

@receiver(post_save, sender=Post)
def maybe_generate_recommendations(sender, instance, created, **kwargs):
    """新規投稿時に一定確率で推薦再計算コマンドをバックグラウンド実行"""
    if not created:
        return

    # 親投稿（初投稿）であれば確実にトリガー
    if instance.parent_post is None and Post.objects.filter(user_id=instance.user_id, parent_post__isnull=True).count() == 1:
        manage_py = os.path.join(settings.BASE_DIR, "manage.py")
        cmd = [sys.executable, manage_py, "generate_recommendations_set", "--sns_type=threadplanet"]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    if random.random() >= TRIGGER_PROB:
        return

    manage_py = os.path.join(settings.BASE_DIR, "manage.py")
    cmd = [sys.executable, manage_py, "generate_recommendations_set", "--sns_type=threadplanet"]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 