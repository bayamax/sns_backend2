import os
from pathlib import Path

# プロジェクトのルートディレクトリパスを設定
BASE_DIR = Path(__file__).resolve().parent.parent





DEBUG = True
ALLOWED_HOSTS = ['178.128.84.239', 'localhost', '127.0.0.1', 'api.daenishi.org']
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}

# URLの設定
ROOT_URLCONF = 'sns_backend.urls'

# 管理画面用のアプリを追加
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.apple',
    'rest_framework',
    'corsheaders',
    'accounts',
    'posts',
    'recommendations',
    'notifications',
    'api',
]

# セキュリティキーの追加
SECRET_KEY = 'django-insecure-your-secret-key-here'

# ミドルウェアの設定
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'allauth.account.middleware.AccountMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# 静的ファイルの設定
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# テンプレート設定
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# データベース設定
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'postgres'),
        'USER': os.environ.get('POSTGRES_USER', 'postgres'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
        'HOST': 'db',
        'PORT': 5432,
    }
}

# カスタムユーザーモデルの設定
AUTH_USER_MODEL = 'accounts.User'

# CORS設定
CORS_ALLOW_ALL_ORIGINS = True  # 開発用。本番環境では特定のオリジンのみ許可するべき

# HTTPS設定
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = False  # Nginxで既にリダイレクトしているため
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True


REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    # 'TRAILING_SLASH': False # ← コメントアウトまたは削除 (デフォルト True)
    'TRAILING_SLASH': True # 明示的にTrueにする場合
}

# --- OpenAI API Key Configuration ---
# 環境変数 'OPENAI_API_KEY' からキーを読み込みます。
# サーバー環境でこの環境変数を設定してください。
# 例: export OPENAI_API_KEY='your_actual_openai_api_key_here'
# もし環境変数がない場合は None となり、各コマンドでエラーまたは警告が出力されます。
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None) # ここにキーを設定。サーバーで編集する場合は 'your_actual_openai_api_key_here' のようなプレースホルダを直接書き換えてもOK

# RECOMMENDATION_MODEL_PATHS = { # 必要であればモデルパスもここに集約
#     'DCOR_AVG_TO_ACCOUNT_MODEL': 'recommendations/pretrained/dcor_filtered_avg_to_account_model.pt',
#     'PROBABILISTIC_FOLLOW_MODEL': 'recommendations/pretrained/probabilistic_followee_model.pt',
# }

# django-allauth 用追加設定
# SITE_ID は django.contrib.sites に必要
SITE_ID = 1

# 認証バックエンドに allauth を追加
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',  # デフォルト
    'allauth.account.auth_backends.AuthenticationBackend',
]

# allauth オプション
ACCOUNT_EMAIL_REQUIRED = False
ACCOUNT_USERNAME_REQUIRED = True
ACCOUNT_AUTHENTICATION_METHOD = 'username'
ACCOUNT_EMAIL_VERIFICATION = 'none'

# Apple プロバイダの設定
_APPLE_PRIVATE_KEY = os.environ.get("APPLE_PRIVATE_KEY")
# 環境変数では改行を "\n" で表現している場合があるので実際の改行に戻す
if _APPLE_PRIVATE_KEY and "\n" in _APPLE_PRIVATE_KEY:
    _APPLE_PRIVATE_KEY = _APPLE_PRIVATE_KEY.replace("\\n", "\n")

SOCIALACCOUNT_PROVIDERS = {
    "apple": {
        "APP": {
            "client_id": os.environ.get("APPLE_CLIENT_ID"),
            "team_id": os.environ.get("APPLE_TEAM_ID"),
            "key": os.environ.get("APPLE_KEY_ID"),
            "secret": _APPLE_PRIVATE_KEY,
        }
    }
}

