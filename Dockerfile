FROM python:3.11-slim

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 静的ファイルの収集
RUN python manage.py collectstatic --noinput

# 公開ポート
EXPOSE 8000

# cronの設定
RUN touch /var/log/cron.log && \
    chmod 0644 /var/log/cron.log

# デフォルトのコマンド（webサービス用）
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "sns_backend.wsgi:application"] 