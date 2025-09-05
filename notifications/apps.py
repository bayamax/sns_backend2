from django.apps import AppConfig

class NotificationsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'notifications'

    def ready(self):
        import logging
        logging.getLogger(__name__).info("### NotificationsConfig.ready CALLED ###")
        import notifications.utils  # シグナルを登録するために通知ユーティリティをインポート