import feedparser
import numpy as np
import torch
import openai # openai v0.28 を想定
import time
import re
import random # ★ random モジュールをインポート
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Q
from sklearn.metrics.pairwise import cosine_similarity

from posts.models import Post
from recommendations.models import UserEmbedding
from recommendations.ml_models import EmbeddingConverter

User = get_user_model()

# (BOT_ACCOUNT_USERNAMES list remains the same)
BOT_ACCOUNT_USERNAMES = [
    "AIStartupDX", "ArcadeVSJP", "ArtExhibitJP", "BaseballHRJP", "BGArt_JP",
    "BirdGardenJP", "BizBookJP", "BizBookJP", "BizBookJP", "CarEnthusiastJP",
    "CasualTalkJP", "CertAIBizJP", "CoffeeCraftJP", "CompCodingJP", "ConservDXJP",
    "CosplayLiveJP", "CraftBeerJP", "DailyChatJP", "DailyVlogJP", "DartsBarJP",
    "DevPatternsJP", "EVMobilityJP", "FIRECareerJP", "FrontierTechJP", "GamerLogJP",
    "GenArtFeed", "GiveawayJP", "HistoryIFJP", "HRCareerJP", "HRWellbeingJP",
    "HYBEKpopJP", "IdolFanLogJP", "IdolFansJP", "IndieDevMaker", "IndieGameApp",
    "IndieMobileAI", "InsectBreedJP", "JLeagueHub", "JPComedyTV", "KenpoPolitics",
    "KimonoDIY", "LiveBandJP", "LocalDXJP", "LocalMuseumJP", "LocalTravelJP",
    "LoveMarryJP", "MangaPanelsJP", "MetaverseVT", "MonozukuriNet", "ObsoleteMedia",
    "OjouRoleJP", "OSSProdDev", "PortraitEvtJP", "PPTDesignLab", "PrizeWinJP",
    "ProgCareerJP", "ProVolleyJP", "RamenReportJP", "ReptileTubeJP", "RetailFastJP",
    "RhythmGameJP", "RPA_LLM", "SeiyuAudioJP", "SentoSaunaJP", "ShukatsuLab",
    "ShuroSupport", "SocBlogJP", "SoundMakersJP", "SpaceWatchJP", "StageLiveJP",
    "SubcultureJP", "TicketTradeJP", "ToyamaLocalJP", "TriviaPromoJP", "USRightWatch",
    "WatchManiaJP", "YouTubeSceneJP"
]

# --- ★ 複数のRSSフィードURLをリストで定義 ---
# (URLは例です。有効なものを追加・修正してください)
RSS_FEED_URL_LIST = [
    'https://www.nhk.or.jp/rss/news/cat0.xml', # NHK主要
    'https://news.yahoo.co.jp/rss/topics/top-picks.xml', # Yahoo!主要
    'https://www.asahi.com/rss/asahi/newsheadlines.rdf', # 朝日新聞
    'https://www.yomiuri.co.jp/rss/yol/latestnews', # 読売新聞
    'https://mainichi.jp/rss/etc/mainichi-flash.rss', # 毎日新聞
    'https://rss.itmedia.co.jp/rss/2.0/itmedia_news.xml', # ITmedia News
    'https://www.gizmodo.jp/index.xml', # Gizmodo Japan
    # 必要に応じて他のURLを追加
]
# -------------------------------------------

# --- 設定値 ---
# RSS_FEED_URL = '...' # ← 直接指定していた行を削除またはコメントアウト
# OPENAI_API_KEY は __init__ でハードコード
OPENAI_EMBEDDING_MODEL = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
POST_MAX_LENGTH = getattr(settings, 'BOT_SETTINGS', {}).get('POST_MAX_LENGTH', 280)
POST_IDENTIFIER_PREFIX = getattr(settings, 'BOT_SETTINGS', {}).get('POST_IDENTIFIER_PREFIX', '【Bot News】')
OPENAI_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('OPENAI_DIM', 3072)
NODE2VEC_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('NODE2VEC_DIM', 128)
CONVERTER_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('CONVERTER_HIDDEN_DIM', 1024)
CONVERTER_MODEL_PATH = getattr(settings, 'MODEL_PATHS', {}).get('EMBEDDING_CONVERTER', 'recommendations/pretrained/openai_to_node2vec_model.pt')
PROCESS_ITEM_LIMIT = getattr(settings, 'BOT_SETTINGS', {}).get('PROCESS_ITEM_LIMIT', 10)
# ----------------------------------------------------


class Command(BaseCommand):
    help = 'Fetches RSS summaries from a randomly selected feed, finds relevant bot, posts as that bot.'

    # (他のメソッド __init__, get_device, etc. は変更なし)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.get_device()
        self.converter_model = self.load_converter_model()
        hardcoded_api_key = "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"
        if hardcoded_api_key:
            openai.api_key = hardcoded_api_key
            self.openai_available = True
        else:
            self.stdout.write(self.style.ERROR("OpenAI API key is missing in the hardcoded variable."))
            self.openai_available = False
        self.bot_users = self.get_bot_user_objects()
        if not self.bot_users:
             self.stdout.write(self.style.ERROR("No valid bot accounts found in the database from the predefined list. Cannot proceed."))

    def get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.zeros(1, device="mps"); return torch.device("mps")
            except Exception: pass
        return torch.device("cpu")

    def load_converter_model(self):
        model = EmbeddingConverter(OPENAI_DIM, NODE2VEC_DIM, CONVERTER_HIDDEN_DIM)
        try:
            model.load_state_dict(torch.load(CONVERTER_MODEL_PATH, map_location=self.device)); model = model.to(self.device); model.eval()
            self.stdout.write(self.style.SUCCESS(f"Converter model loaded from {CONVERTER_MODEL_PATH}"))
            return model
        except FileNotFoundError: self.stdout.write(self.style.ERROR(f"Converter model not found at {CONVERTER_MODEL_PATH}.")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading converter model: {e}")); return None

    def get_bot_user_objects(self):
        unique_bot_usernames = list(set(BOT_ACCOUNT_USERNAMES))
        users = User.objects.filter(username__in=unique_bot_usernames)
        found_usernames = [user.username for user in users]
        missing_usernames = [name for name in unique_bot_usernames if name not in found_usernames]
        if missing_usernames: self.stdout.write(self.style.WARNING(f"Bots not found: {', '.join(missing_usernames)}"))
        self.stdout.write(f"Found {users.count()} valid bot accounts.")
        return list(users)

    def fetch_rss_summaries(self, url): # 引数 url を受け取るのはそのまま
        self.stdout.write(f"Fetching summaries from RSS feed: {url}") # どのURLかログ表示
        summary_list = []
        try:
            feed = feedparser.parse(url)
            if feed.bozo: self.stdout.write(self.style.WARNING(f"Error parsing RSS feed: {feed.bozo_exception}")); return []
            for entry in feed.entries:
                title = getattr(entry, 'title', None); link = getattr(entry, 'link', None); summary = getattr(entry, 'summary', getattr(entry, 'description', None))
                if title and link and summary: summary_list.append({'title': title, 'link': link, 'summary': summary})
            self.stdout.write(f"Fetched {len(summary_list)} items.")
            return summary_list
        except Exception as e: self.stdout.write(self.style.ERROR(f"Failed to fetch/parse RSS: {e}")); return []

    def get_openai_embedding(self, text):
        if not self.openai_available: return None
        if not text: return None
        cleaned_text = re.sub('<[^<]+?>', '', text).strip()
        if not cleaned_text: return None
        try:
            response = openai.Embedding.create(input=cleaned_text, model=OPENAI_EMBEDDING_MODEL)
            return np.array(response['data'][0]['embedding'])
        except openai.error.RateLimitError as e: self.stdout.write(self.style.WARNING(f"OpenAI Rate limit: {e}. Waiting...")); time.sleep(20); return None
        except openai.error.InvalidRequestError as e: self.stdout.write(self.style.WARNING(f"OpenAI Invalid Request: {e}")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error getting OpenAI embedding: {e}")); return None

    def convert_to_node2vec(self, openai_vector):
        if self.converter_model is None or openai_vector is None: return None
        if openai_vector.shape[0] != OPENAI_DIM: self.stdout.write(self.style.ERROR(f"Input vector dim mismatch.")); return None
        try:
            with torch.no_grad():
                tensor = torch.tensor(openai_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                output = self.converter_model(tensor)
                return output.squeeze(0).cpu().numpy()
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error converting vector: {e}")); return None

    def find_most_similar_bot(self, target_vector, bot_user_objects):
        if target_vector is None or not bot_user_objects: return None
        if target_vector.shape[0] != NODE2VEC_DIM: self.stdout.write(self.style.ERROR(f"Target vector dim mismatch.")); return None
        bot_usernames = [user.username for user in bot_user_objects]
        bot_embeddings = UserEmbedding.objects.filter(user__username__in=bot_usernames, node2vec_vector__isnull=False).select_related('user')
        if not bot_embeddings.exists(): self.stdout.write(self.style.WARNING("No embeddings for bot accounts.")); return None
        bot_vectors = []; bot_embedding_users = []
        for be in bot_embeddings:
            try:
                vec = np.array(be.node2vec_vector)
                if vec.shape == (NODE2VEC_DIM,): bot_vectors.append(vec); bot_embedding_users.append(be.user)
            except (TypeError, ValueError): pass
        if not bot_vectors: self.stdout.write(self.style.WARNING("No valid vectors for bots.")); return None
        try:
            target_vec_reshaped = target_vector.reshape(1, -1)
            if target_vec_reshaped.shape[1] != NODE2VEC_DIM: self.stdout.write(self.style.ERROR("Reshaped vector dim incorrect.")); return None
            similarity_scores = cosine_similarity(target_vec_reshaped, np.array(bot_vectors))
            most_similar_index = np.argmax(similarity_scores[0])
            most_similar_bot_user = bot_embedding_users[most_similar_index]
            self.stdout.write(f"Most similar bot: {most_similar_bot_user.username}")
            return most_similar_bot_user
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error finding similar bot: {e}")); return None

    def create_post_as_similar_bot(self, post_author_user, content_summary, original_link):
        cleaned_summary = re.sub('<[^<]+?>', '', content_summary).strip()
        if not cleaned_summary: return None
        post_content = f"{POST_IDENTIFIER_PREFIX}\n\"{cleaned_summary}\""
        if len(post_content) > POST_MAX_LENGTH:
             available_length = POST_MAX_LENGTH - len(f"{POST_IDENTIFIER_PREFIX}\n\"\"") - 3
             if available_length < 10: self.stdout.write(self.style.WARNING(f"Cannot fit summary.")); return None
             truncated_summary = cleaned_summary[:available_length] + "..."
             post_content = f"{POST_IDENTIFIER_PREFIX}\n\"{truncated_summary}\""
        if len(post_content) > POST_MAX_LENGTH: self.stdout.write(self.style.WARNING(f"Final post too long.")); return None
        try:
            # 同じ元記事リンクの投稿が既にないかチェック
            if Post.objects.filter(content__icontains=original_link).exists():
                 self.stdout.write(self.style.WARNING(f"Post link exists: {original_link}"))
                 return None
            post = Post.objects.create(user=post_author_user, content=post_content)
            self.stdout.write(self.style.SUCCESS(f"Created post ID {post.id} as {post_author_user.username}"))
            return post
        except Exception as e: self.stdout.write(self.style.ERROR(f"Failed post as {post_author_user.username}: {e}")); return None

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting bot posting process (Random Feed, Post as similar bot)...")) # help メッセージを更新

        if not self.openai_available: self.stdout.write(self.style.ERROR("OpenAI key not available.")); return
        if not self.converter_model: self.stdout.write(self.style.ERROR("Converter model not loaded.")); return
        if not self.bot_users: self.stdout.write(self.style.ERROR("No valid bot accounts.")); return

        # --- ★ ここでRSSフィードURLをランダムに選択 ---
        if not RSS_FEED_URL_LIST:
             self.stdout.write(self.style.ERROR("RSS_FEED_URL_LIST is empty. Cannot select a feed."))
             return
        selected_rss_url = random.choice(RSS_FEED_URL_LIST)
        self.stdout.write(f"Selected RSS feed: {selected_rss_url}")
        # -----------------------------------------

        # 選択されたURLを使ってニュースを取得
        content_items = self.fetch_rss_summaries(selected_rss_url) # ★ 選択したURLを渡す
        if not content_items:
            self.stdout.write(self.style.WARNING(f"No content items fetched from {selected_rss_url}. Exiting."))
            return

        processed_count = 0
        posted_count = 0
        items_to_process = content_items[:PROCESS_ITEM_LIMIT]
        self.stdout.write(f"Attempting to process up to {len(items_to_process)} items...")

        for item in items_to_process:
            link = item.get('link')
            title = item.get('title')
            summary = item.get('summary')
            if not link or not title or not summary: continue

            processed_count += 1
            self.stdout.write(f"\n[{processed_count}/{len(items_to_process)}] Processing: {title}")

            cleaned_summary = re.sub('<[^<]+?>', '', summary).strip()
            if len(cleaned_summary) == 0: self.stdout.write(self.style.WARNING(f"Skipping '{title}': Empty summary.")); continue

            openai_embedding = self.get_openai_embedding(cleaned_summary)
            if openai_embedding is None: time.sleep(1); continue

            node2vec_embedding = self.convert_to_node2vec(openai_embedding)
            if node2vec_embedding is None: time.sleep(1); continue

            similar_bot_user = self.find_most_similar_bot(node2vec_embedding, self.bot_users)
            if similar_bot_user is None: self.stdout.write(self.style.WARNING(f"No similar bot for: {title}")); time.sleep(1); continue

            # ★ 重複チェック用に link も渡すように修正
            created_post = self.create_post_as_similar_bot(similar_bot_user, summary, link)
            if created_post:
                posted_count += 1
                self.stdout.write("Waiting...")
                time.sleep(3)
            else:
                time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"\nFinished. Processed {processed_count}, created {posted_count} posts."))