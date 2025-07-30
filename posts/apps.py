from django.apps import AppConfig


class PostsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'posts'

    def ready(self):
        # 投稿保存後シグナルを登録
        import posts.signals  # noqa: F401
