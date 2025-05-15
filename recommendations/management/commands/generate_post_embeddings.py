# recommendations/management/commands/generate_post_embeddings.py

import os
import openai
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db.models import Q
from posts.models import Post
from recommendations.models import PostEmbedding

class Command(BaseCommand):
    help = 'Generate OpenAI embeddings for posts'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of embeddings for posts that already have them'
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit the number of posts to process'
        )

    def handle(self, *args, **options):
        force = options.get('force', False)
        limit = options.get('limit')
        
        # APIキーをsettingsから取得するように修正
        # openai.api_key = "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"
        api_key_from_settings = getattr(settings, 'OPENAI_API_KEY', None)
        if api_key_from_settings:
            openai.api_key = api_key_from_settings
        else:
            self.stdout.write(self.style.ERROR("OpenAI API key not found in Django settings or environment variables. Cannot proceed."))
            return # APIキーがない場合は処理を中断
        
        # 埋め込みベクトルがまだないポストを取得
        if force:
            posts = Post.objects.all()
        else:
            posts = Post.objects.filter(~Q(embedding__isnull=False))
        
        if limit:
            posts = posts[:limit]
        
        self.stdout.write(f"Processing {posts.count()} posts")
        
        # 各ポストに対して埋め込みベクトルを生成
        for post in posts:
            try:
                # OpenAI APIを使用して埋め込みを取得
                response = openai.Embedding.create(
                    input=post.content,
                    model="text-embedding-3-large"
                )
                vector = response['data'][0]['embedding']
                
                # 埋め込みを保存
                embedding, created = PostEmbedding.objects.update_or_create(
                    post=post,
                    defaults={'vector': vector}
                )
                
                self.stdout.write(f"Post ID {post.id}: Embedding {'created' if created else 'updated'}")
            
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing post ID {post.id}: {str(e)}"))