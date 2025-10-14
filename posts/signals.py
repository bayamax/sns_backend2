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

    # 初回投稿は確定でトリガー
    if Post.objects.filter(user_id=instance.user_id, parent__isnull=True).count() == 1:
        _trigger_recommendations_job()
        return

    if random.random() >= TRIGGER_PROB:
        return

    _trigger_recommendations_job()

def _trigger_recommendations_job():
    manage_py = os.path.join(settings.BASE_DIR, "manage.py")
    cmd = [sys.executable, manage_py, "generate_recommendations_set", "--sns_type=threadplanet"]
    # バックグラウンドで実行（標準出力・エラーは捨てるかログに流す）
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 