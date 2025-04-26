from django.core.management.base import BaseCommand
from django.core.management import call_command # Djangoコマンド呼び出し用
import time
import logging
from django.utils import timezone
# 不要になったインポートをコメントアウトまたは削除
# from recommendations.models import UserRecommendation, UserEmbedding, PostEmbedding
# from accounts.models import User
# from django.db.models import Count

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs a scheduler to execute the update_recommendations command daily at midnight' # helpメッセージを更新

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting recommendation update scheduler'))
        
        while True:
            now = timezone.now()
            # 次の実行時刻 (次の日の午前0時)
            tomorrow = now.date() + timezone.timedelta(days=1)
            next_run_time = timezone.make_aware(timezone.datetime.combine(tomorrow, timezone.time.min))
            
            # 現在時刻がすでに次の実行時刻を過ぎている場合（起動直後など）は、
            # さらに次の日（つまり明後日）の0時を設定
            if now >= next_run_time:
                next_run_time += timezone.timedelta(days=1)

            wait_seconds = (next_run_time - now).total_seconds()
            
            self.stdout.write(f'Current time: {now}. Next run scheduled at: {next_run_time}. Waiting for {wait_seconds:.0f} seconds.')
            time.sleep(wait_seconds)

            # --- 定刻になったのでコマンドを実行 --- 
            self.stdout.write(self.style.SUCCESS(f'It is now {timezone.now()}. Running recommendation update pipeline...'))
            try:
                # update_recommendations コマンドを実行
                # ここではオプションなしで全ユーザー対象に実行
                call_command('update_recommendations') 
                self.stdout.write(self.style.SUCCESS('Recommendation update pipeline finished successfully'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error running update_recommendations command: {e}'))
                # エラーログに詳細を出力
                import traceback
                logger.error(f"Error in scheduler during update_recommendations: {traceback.format_exc()}")
            # --- コマンド実行終了 --- 

    # --- 既存の generate_recommendations メソッドは不要になったため削除 --- 
    # def generate_recommendations(self):
    #     """ユーザー推薦を生成する (旧ロジック)"""
    #     # ... (このメソッドの内容は削除) ...
    #     pass 
    # ---------------------------------------------------------------------
