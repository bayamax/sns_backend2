import feedparser
import numpy as np
import torch
import openai # openai v0.28 を想定
import time
import re
from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Q # Qオブジェクトをインポート
from sklearn.metrics.pairwise import cosine_similarity
# newspaper3k は不要になったので削除
# import requests も直接は使わないので削除可能

from posts.models import Post
from recommendations.models import UserEmbedding
from recommendations.ml_models import EmbeddingConverter

User = get_user_model()

# --- 固定のボットアカウントリスト ---
# 注: リスト内に重複がありますが、User.objects.filter(username__in=...) で処理可能です
BOT_ACCOUNT_USERNAMES = [
    "AIStartupDX", "ArcadeVSJP", "ArtExhibitJP", "BaseballHRJP", "BGArt_JP",
    "BirdGardenJP", "BizBookJP", "BizBookJP", "BizBookJP", "CarEnthusiastJP", # BizBookJPの重複はそのまま
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
# ----------------------------------

# --- 設定値 (settings.py や環境変数から取得推奨だが、上記リストを使うためBOT_USERNAMEは削除) ---
RSS_FEED_URL = getattr(settings, 'BOT_SETTINGS', {}).get('RSS_FEED_URL', 'http://feeds.bbci.co.uk/news/rss.xml')
OPENAI_API_KEY = getattr(settings, 'OPENAI_API_KEY', None)
# ★ 既存システムで使用されているモデルに合わせる
OPENAI_EMBEDDING_MODEL = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large') # 修正
# 投稿文字数制限 (Postモデルの max_length 等に合わせて調整してください)
POST_MAX_LENGTH = getattr(settings, 'BOT_SETTINGS', {}).get('POST_MAX_LENGTH', 280) # 例: 280文字
POST_IDENTIFIER_PREFIX = getattr(settings, 'BOT_SETTINGS', {}).get('POST_IDENTIFIER_PREFIX', '【Bot News】') # Prefixは残すか確認
# ★ text-embedding-3-large の次元数に合わせる
OPENAI_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('OPENAI_DIM', 3072) # 修正
NODE2VEC_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('NODE2VEC_DIM', 128)
CONVERTER_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('CONVERTER_HIDDEN_DIM', 1024)
CONVERTER_MODEL_PATH = getattr(settings, 'MODEL_PATHS', {}).get('EMBEDDING_CONVERTER', 'recommendations/pretrained/openai_to_node2vec_model.pt')
PROCESS_ITEM_LIMIT = getattr(settings, 'BOT_SETTINGS', {}).get('PROCESS_ITEM_LIMIT', 10) # 一度に試行する記事数
# ----------------------------------------------------

class Command(BaseCommand):
    help = 'Fetches RSS summaries, finds the most relevant bot account from a fixed list, and posts the summary as that bot.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.get_device()
        self.converter_model = self.load_converter_model()
        if OPENAI_API_KEY:
            # openai v0.28 の初期化方法
            openai.api_key = OPENAI_API_KEY
            self.openai_available = True
        else:
            self.stdout.write(self.style.ERROR("OpenAI API key not configured in settings."))
            self.openai_available = False
        # ボットアカウントのUserオブジェクトリストを初期化時に取得・保持
        self.bot_users = self.get_bot_user_objects()
        if not self.bot_users:
             # 致命的エラーとするか、警告にとどめるか
             self.stdout.write(self.style.ERROR("No valid bot accounts found in the database from the predefined list. Cannot proceed."))
             # self.bot_users を None にして後続処理をスキップさせるなども可能
             # exit(1) # またはプログラムを終了させる

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")

    def load_converter_model(self):
        # Note: Converter モデルが想定する入力次元(OPENAI_DIM)が、
        # 使用するEmbeddingモデル(ada-002なら1536)と一致しているか確認してください。
        model = EmbeddingConverter(OPENAI_DIM, NODE2VEC_DIM, CONVERTER_HIDDEN_DIM)
        try:
            # モデルパスが settings.BASE_DIR からの相対パスか絶対パスか確認
            # model_path = os.path.join(settings.BASE_DIR, CONVERTER_MODEL_PATH) などが必要かも
            model.load_state_dict(torch.load(CONVERTER_MODEL_PATH, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.stdout.write(self.style.SUCCESS(f"Converter model loaded successfully from {CONVERTER_MODEL_PATH}"))
            return model
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f"Converter model not found at {CONVERTER_MODEL_PATH}. Ensure the path is correct."))
            return None # モデルがない場合は処理を続けられないので None
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading converter model: {e}"))
            return None

    def get_bot_user_objects(self):
        """定義済みリストから存在するボットユーザーオブジェクトを取得"""
        # 重複を除去したリストでクエリ
        unique_bot_usernames = list(set(BOT_ACCOUNT_USERNAMES))
        users = User.objects.filter(username__in=unique_bot_usernames)
        found_usernames = [user.username for user in users]
        missing_usernames = [name for name in unique_bot_usernames if name not in found_usernames]
        if missing_usernames:
            self.stdout.write(self.style.WARNING(f"The following bot accounts were not found in the database: {', '.join(missing_usernames)}"))
        self.stdout.write(f"Found {users.count()} valid bot accounts to use for posting.")
        return list(users) # リストとして返す

    def fetch_rss_summaries(self, url):
        """RSSフィードから記事のタイトル、リンク、要約を取得"""
        self.stdout.write(f"Fetching summaries from RSS feed: {url}")
        summary_list = []
        try:
            feed = feedparser.parse(url)
            if feed.bozo:
                self.stdout.write(self.style.WARNING(f"Error parsing RSS feed: {feed.bozo_exception}"))
                return []

            for entry in feed.entries:
                # title, link, summary (または description) が存在するかチェック
                title = getattr(entry, 'title', None)
                link = getattr(entry, 'link', None)
                summary = getattr(entry, 'summary', getattr(entry, 'description', None)) # summary がなければ description を試す

                if title and link and summary:
                    summary_list.append({
                        'title': title,
                        'link': link,
                        'summary': summary # HTMLタグが含まれる場合があるので注意
                    })
            self.stdout.write(f"Fetched {len(summary_list)} items with summaries from RSS feed.")
            return summary_list
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to fetch or parse RSS feed: {e}"))
            return []

    def get_openai_embedding(self, text):
        """OpenAI APIを使用して埋め込みベクトルを取得 (v0.28 形式)"""
        if not self.openai_available:
            return None
        if not text:
            return None
        # 簡単なHTMLタグ除去
        cleaned_text = re.sub('<[^<]+?>', '', text).strip()
        if not cleaned_text:
             return None

        try:
            # ★ モデル名を OPENAI_EMBEDDING_MODEL 定数から取得するように修正
            response = openai.Embedding.create(
                input=cleaned_text, # クリーンなテキストを使用
                model=OPENAI_EMBEDDING_MODEL
            )
            return np.array(response['data'][0]['embedding'])
        except openai.error.RateLimitError as e:
            self.stdout.write(self.style.WARNING(f"OpenAI Rate limit exceeded for embedding: {e}. Waiting..."))
            time.sleep(20)
            return None
        except openai.error.InvalidRequestError as e:
             self.stdout.write(self.style.WARNING(f"OpenAI Invalid Request for embedding (maybe text too long?): {e}"))
             return None
        except Exception as e:
             self.stdout.write(self.style.ERROR(f"Error getting OpenAI embedding for text starting with '{cleaned_text[:50]}...': {e}"))
        return None

    def convert_to_node2vec(self, openai_vector):
        if self.converter_model is None or openai_vector is None:
            return None
        if openai_vector.shape[0] != OPENAI_DIM:
             self.stdout.write(self.style.ERROR(f"Input vector dimension ({openai_vector.shape[0]}) != OPENAI_DIM ({OPENAI_DIM})."))
             return None
        try:
            with torch.no_grad():
                tensor = torch.tensor(openai_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                output = self.converter_model(tensor)
                return output.squeeze(0).cpu().numpy()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error converting vector: {e}"))
            return None

    def find_most_similar_bot(self, target_vector, bot_user_objects):
        """指定されたベクトルに最も類似した「ボットアカウント」を検索"""
        if target_vector is None or not bot_user_objects:
            return None
        if target_vector.shape[0] != NODE2VEC_DIM:
             self.stdout.write(self.style.ERROR(f"Target vector dim ({target_vector.shape[0]}) != NODE2VEC_DIM ({NODE2VEC_DIM})."))
             return None

        # ボットアカウントの UserEmbedding を取得
        # 注意: ボットアカウントにも UserEmbedding が事前に計算されている必要がある
        bot_usernames = [user.username for user in bot_user_objects]
        bot_embeddings = UserEmbedding.objects.filter(
            user__username__in=bot_usernames,
            node2vec_vector__isnull=False
        ).select_related('user')

        if not bot_embeddings.exists():
            self.stdout.write(self.style.WARNING("No embeddings found for any of the specified bot accounts."))
            return None

        bot_vectors = []
        bot_embedding_users = [] # UserEmbedding に対応する User オブジェクト
        for be in bot_embeddings:
            try:
                vec = np.array(be.node2vec_vector)
                if vec.shape == (NODE2VEC_DIM,):
                     bot_vectors.append(vec)
                     bot_embedding_users.append(be.user) # UserEmbedding の user を使う
            except (TypeError, ValueError):
                 pass # 不正なベクトルはスキップ

        if not bot_vectors:
             self.stdout.write(self.style.WARNING("No valid vectors found for bot accounts."))
             return None

        try:
            target_vec_reshaped = target_vector.reshape(1, -1)
            if target_vec_reshaped.shape[1] != NODE2VEC_DIM:
                 self.stdout.write(self.style.ERROR("Reshaped target vector dimension is incorrect."))
                 return None

            # ニュースベクトルと「ボットアカウントのベクトル」で類似度計算
            similarity_scores = cosine_similarity(target_vec_reshaped, np.array(bot_vectors))
            most_similar_index = np.argmax(similarity_scores[0])
            most_similar_bot_user = bot_embedding_users[most_similar_index] # 最も類似したボットのUserオブジェクト
            self.stdout.write(f"Most similar bot account for this content: {most_similar_bot_user.username}")
            return most_similar_bot_user # User オブジェクトを返す
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during bot similarity calculation: {e}"))
            return None

    def create_post_as_similar_bot(self, post_author_user, content_summary, original_link):
        """指定されたボットアカウントを投稿者として投稿を作成"""
        cleaned_summary = re.sub('<[^<]+?>', '', content_summary).strip()
        if not cleaned_summary:
            return None

        # 投稿内容は Prefix + 要約のみ
        post_content = f"{POST_IDENTIFIER_PREFIX}\n\"{cleaned_summary}\""

        # 文字数制限チェック (Prefix + 要約)
        if len(post_content) > POST_MAX_LENGTH:
             # 要約を切り詰める
             available_length = POST_MAX_LENGTH - len(f"{POST_IDENTIFIER_PREFIX}\n\"\"") - 3 # 余裕
             if available_length < 10:
                  self.stdout.write(self.style.WARNING(f"Cannot fit summary within {POST_MAX_LENGTH} chars even after truncation attempt."))
                  return None
             truncated_summary = cleaned_summary[:available_length] + "..."
             post_content = f"{POST_IDENTIFIER_PREFIX}\n\"{truncated_summary}\""

        # 最終チェック
        if len(post_content) > POST_MAX_LENGTH:
             self.stdout.write(self.style.WARNING(f"Final post content still exceeds {POST_MAX_LENGTH}. Skipping."))
             return None

        try:
            # 重複チェック: 同じボットが同じ内容を投稿していないか？
            # または、同じ元記事リンクの投稿が既にないか？ -> link を使う方が安全か？
            # ここでは original_link を使って重複チェックを行う
            if Post.objects.filter(content__icontains=original_link).exists():
                 self.stdout.write(self.style.WARNING(f"Post with original link '{original_link}' seems to exist already. Skipping."))
                 return None
            # あるいは、同じボットが同じ内容を投稿していないかチェック
            # if Post.objects.filter(user=post_author_user, content=post_content).exists():
            #     return None

            post = Post.objects.create(
                user=post_author_user, # ★ 投稿者を類似ボットに設定
                content=post_content
            )
            self.stdout.write(self.style.SUCCESS(f"Created post ID {post.id} as user '{post_author_user.username}'"))
            return post
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to create post as {post_author_user.username}: {e}"))
            return None

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting bot posting process (Post as most similar bot)..."))

        if not self.openai_available:
            self.stdout.write(self.style.ERROR("OpenAI API key not configured. Exiting."))
            return
        if not self.converter_model:
             self.stdout.write(self.style.ERROR("Embedding converter model not loaded. Exiting."))
             return
        if not self.bot_users: # __init__で取得したボットリストを確認
             self.stdout.write(self.style.ERROR("No valid bot accounts available for posting. Exiting."))
             return

        # 1. RSSから要約リストを取得
        content_items = self.fetch_rss_summaries(RSS_FEED_URL)
        if not content_items:
            self.stdout.write(self.style.WARNING("No content items with summaries fetched. Exiting."))
            return

        processed_count = 0
        posted_count = 0
        items_to_process = content_items[:PROCESS_ITEM_LIMIT]
        self.stdout.write(f"Attempting to process up to {len(items_to_process)} items...")

        for item in items_to_process:
            link = item.get('link') # 重複チェック用に使う
            title = item.get('title') # ログ表示用に使う
            summary = item.get('summary')

            if not link or not title or not summary:
                continue

            processed_count += 1
            self.stdout.write(f"\n[{processed_count}/{len(items_to_process)}] Processing: {title}")

            # --- ステップ2: (文字数チェックは投稿作成時に行う) ---
            # HTMLを除去した要約を取得
            cleaned_summary = re.sub('<[^<]+?>', '', summary).strip()
            if len(cleaned_summary) == 0:
                self.stdout.write(self.style.WARNING(f"Skipping '{title}': Summary empty after cleaning."))
                continue

            # --- ステップ3: OpenAI Embedding取得 ---
            openai_embedding = self.get_openai_embedding(cleaned_summary) # クリーンな要約を使う
            if openai_embedding is None:
                time.sleep(1)
                continue

            # --- ステップ4: ベクトル変換 ---
            node2vec_embedding = self.convert_to_node2vec(openai_embedding)
            if node2vec_embedding is None:
                 time.sleep(1)
                 continue

            # --- ステップ5: 最も類似した「ボットアカウント」を検索 ---
            # self.bot_users (初期化時に取得したリスト) を渡す
            similar_bot_user = self.find_most_similar_bot(node2vec_embedding, self.bot_users)
            if similar_bot_user is None:
                # find_most_similar_bot 内でログが出るはず
                self.stdout.write(self.style.WARNING(f"Could not find a similar bot account for: {title}"))
                time.sleep(1)
                continue

            # --- ステップ6: 類似ボットアカウントとして投稿を作成 ---
            # 引数を変更: bot_user -> similar_bot_user, content_summary, original_link (重複チェック用)
            created_post = self.create_post_as_similar_bot(similar_bot_user, summary, link)
            if created_post:
                posted_count += 1
                self.stdout.write("Waiting before processing next item...")
                time.sleep(3)
            else:
                # 投稿失敗時 (重複含む、文字数超過など)
                time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"\nBot posting process finished. Processed {processed_count} items, created {posted_count} posts."))
