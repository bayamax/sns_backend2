import os
from pathlib import Path

# プロジェクトのルートディレクトリパスを設定
BASE_DIR = Path(__file__).resolve().parent.parent





DEBUG = True
ALLOWED_HOSTS = ['178.128.84.239', 'localhost', '127.0.0.1']
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
    'rest_framework',
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
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
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
INSTALLED_APPS += ['corsheaders']
MIDDLEWARE.insert(1, 'corsheaders.middleware.CorsMiddleware')

# HTTPS設定
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = False  # Nginxで既にリダイレクトしているため

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = False 
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
ALLOWED_HOSTS += ['api.daenishi.org'] 

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = False 
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
ALLOWED_HOSTS += ['api.daenishi.org'] 

# APPEND_SLASH = False # ← コメントアウトまたは削除 (デフォルト True)
APPEND_SLASH = True # 明示的にTrueにする場合

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
