# recommendations/management/commands/create_bot_posts.py

import feedparser
import numpy as np
import torch
import openai # openai v0.28 を想定
import time
import re
import random
import os # osモジュールをインポート
from django.core.management.base import BaseCommand
from django.conf import settings # settingsをインポート
from django.contrib.auth import get_user_model
from django.db.models import Q
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

from posts.models import Post
from recommendations.models import UserEmbedding

User = get_user_model()

# --- 定数 ---
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
RSS_FEED_URL_LIST = [
    'https://www.nhk.or.jp/rss/news/cat0.xml', # NHK主要 (リストはそのまま)
    #'https://www.gizmodo.jp/index.xml',
    #'https://www.asahi.com/rss/asahi/newsheadlines.rdf', # 朝日新聞
    #'https://www.yomiuri.co.jp/rss/yol/latestnews', # 読売新聞
    #'https://mainichi.jp/rss/etc/mainichi-flash.rss', # 毎日新聞
    #'https://rss.itmedia.co.jp/rss/2.0/itmedia_news.xml', 
]

# --- ★ ハードコードする設定値 ★ ---
HARDCODED_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-large'
HARDCODED_POST_MAX_LENGTH = 280
HARDCODED_POST_IDENTIFIER_PREFIX = '【Bot News】'
HARDCODED_OPENAI_DIM = 3072
HARDCODED_NODE2VEC_DIM = 128
# HARDCODED_AVG_POST_TO_ACCOUNT_MODEL_PATH = 'recommendations/pretrained/avg_post_to_account_model.pt' # 古いパスをコメントアウトまたは削除
HARDCODED_DCOR_MODEL_PATH = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'dcor_filtered_avg_to_account_model.pt') # settings.BASE_DIR を使ったパス
HARDCODED_PROCESS_ITEM_LIMIT = 10
# -----------------------------------

# --- ★ ユーザー提供の新しいモデル定義に置き換え ---
# class AvgPostToAccountModel(nn.Module): ... (古いモデル定義を削除)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class DcorFilteredAvgPostToAccountModel(nn.Module):
    # input_dim, output_dim, dropout は __init__ で渡される
    # デフォルトの dropout 値はユーザー提供のコードから参照 (0.2)
    def __init__(self, input_dim=HARDCODED_OPENAI_DIM, output_dim=HARDCODED_NODE2VEC_DIM, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(input_dim, 1024, dropout),
            ResidualBlock(1024, 512, dropout),
            ResidualBlock(512, 256, dropout),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.model(x)
# --------------------------------------------------------------

class Command(BaseCommand):
    help = 'Uses DcorFilteredAvgPostToAccountModel to map news embedding and find similar bot (Hardcoded settings).'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.get_device()

        # --- ★ ハードコードされた設定値をインスタンス変数に設定 ---
        self.openai_embedding_model = HARDCODED_OPENAI_EMBEDDING_MODEL
        self.post_max_length = HARDCODED_POST_MAX_LENGTH
        self.post_identifier_prefix = HARDCODED_POST_IDENTIFIER_PREFIX
        self.openai_dim = HARDCODED_OPENAI_DIM
        self.node2vec_dim = HARDCODED_NODE2VEC_DIM
        # self.avg_post_to_account_model_path = HARDCODED_AVG_POST_TO_ACCOUNT_MODEL_PATH # 古い変数
        self.dcor_model_path = HARDCODED_DCOR_MODEL_PATH # 新しい変数
        self.process_item_limit = HARDCODED_PROCESS_ITEM_LIMIT
        # ----------------------------------------------------------

        # self.avg_post_to_account_model = self.load_avg_post_to_account_model() # 古いモデル読み込み
        self.dcor_filtered_avg_to_account_model = self.load_dcor_filtered_avg_to_account_model() # 新しいモデル読み込み
        hardcoded_api_key = "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"
        if hardcoded_api_key: openai.api_key = hardcoded_api_key; self.openai_available = True
        else: self.stdout.write(self.style.ERROR("OpenAI API key missing.")); self.openai_available = False
        self.bot_users = self.get_bot_user_objects()
        if not self.bot_users: self.stdout.write(self.style.ERROR("No valid bot accounts found."))

    def get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try: torch.zeros(1, device="mps"); return torch.device("mps")
            except Exception: pass
        return torch.device("cpu")

    # ★ モデルロードメソッド - モデルクラスの定義に合わせてインスタンス化
    # def load_avg_post_to_account_model(self): ... (古いメソッドを削除)
    def load_dcor_filtered_avg_to_account_model(self):
        # 正しいクラス (ここで定義した新しい構造) を使う
        # Dropout率はモデル定義のデフォルト値(0.2)が使われるが、もし学習時と異なる値を使いたい場合は、
        # ここで DcorFilteredAvgPostToAccountModel の引数に dropout を指定する必要がある。
        # ユーザー提供の学習コードでは DROPOUT_RATE = 0.2 だったので、デフォルトで問題ないはず。
        model = DcorFilteredAvgPostToAccountModel(input_dim=self.openai_dim, output_dim=self.node2vec_dim)
        try:
            # model.load_state_dict(torch.load(self.avg_post_to_account_model_path, map_location=self.device)) # 古いパス
            model.load_state_dict(torch.load(self.dcor_model_path, map_location=self.device)) # 新しいパス
            model = model.to(self.device)
            model.eval()
            # self.stdout.write(self.style.SUCCESS(f"Loaded AvgPostToAccountModel from {self.avg_post_to_account_model_path}"))
            self.stdout.write(self.style.SUCCESS(f"Loaded DcorFilteredAvgPostToAccountModel from {self.dcor_model_path}"))
            return model
        # except FileNotFoundError: self.stdout.write(self.style.ERROR(f"Model not found: {self.avg_post_to_account_model_path}")); return None
        except FileNotFoundError: self.stdout.write(self.style.ERROR(f"Model not found: {self.dcor_model_path}")); return None
        except RuntimeError as e: # ★ state_dictミスマッチエラーを捕捉
             self.stdout.write(self.style.ERROR(f"Error loading state_dict (structure mismatch): {e}"))
             # self.stdout.write(self.style.WARNING("Ensure the AvgPostToAccountModel definition in create_bot_posts.py matches the saved model structure."))
             self.stdout.write(self.style.WARNING("Ensure the DcorFilteredAvgPostToAccountModel definition in create_bot_posts.py matches the saved model structure."))
             return None
        # except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading AvgPostToAccountModel: {e}")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading DcorFilteredAvgPostToAccountModel: {e}")); return None

    # (get_bot_user_objects, fetch_rss_summaries, get_openai_embedding, map_openai_to_account_vector, find_most_similar_bot, create_post_as_similar_bot, handle unchanged, but use self.* variables)
    def get_bot_user_objects(self):
        unique_bot_usernames = list(set(BOT_ACCOUNT_USERNAMES)); users = User.objects.filter(username__in=unique_bot_usernames)
        found_usernames = [user.username for user in users]; missing_usernames = [name for name in unique_bot_usernames if name not in found_usernames]
        if missing_usernames: self.stdout.write(self.style.WARNING(f"Bots not found: {', '.join(missing_usernames)}"))
        self.stdout.write(f"Found {users.count()} valid bot accounts.")
        return list(users)

    def fetch_rss_summaries(self, url):
        self.stdout.write(f"Fetching summaries from RSS feed: {url}"); summary_list = []
        try:
            feed = feedparser.parse(url)
            if feed.bozo: self.stdout.write(self.style.WARNING(f"Error parsing RSS: {feed.bozo_exception}")); return []
            for entry in feed.entries:
                title = getattr(entry, 'title', None); link = getattr(entry, 'link', None); summary = getattr(entry, 'summary', getattr(entry, 'description', None))
                if title and link and summary: summary_list.append({'title': title, 'link': link, 'summary': summary})
            self.stdout.write(f"Fetched {len(summary_list)} items."); return summary_list
        except Exception as e: self.stdout.write(self.style.ERROR(f"Failed fetch/parse RSS: {e}")); return []

    def get_openai_embedding(self, text):
        # 1. APIキーのチェック
        if not self.openai_available:
            # self.stdout.write(self.style.ERROR("OpenAI API key not available.")) # ログは冗長なので省略可
            return None
        # 2. 入力テキストの基本的なチェック
        if not text:
            return None

        # 3. テキストのクリーニングと代入
        cleaned_text = re.sub('<[^<]+?>', '', text).strip()

        # 4. クリーニング後のテキストが空でないかチェック
        if not cleaned_text:
            # self.stdout.write(self.style.WARNING("Text became empty after cleaning HTML.")) # 必要ならログ出力
            return None

        # 5. OpenAI API 呼び出し
        try:
            response = openai.Embedding.create(input=cleaned_text, model=self.openai_embedding_model)
            return np.array(response['data'][0]['embedding'])
        except openai.error.RateLimitError as e: self.stdout.write(self.style.WARNING(f"OpenAI Rate limit: {e}. Wait...")); time.sleep(20); return None
        except openai.error.InvalidRequestError as e: self.stdout.write(self.style.WARNING(f"OpenAI Invalid Request: {e}")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error get OpenAI embed: {e}")); return None

    def map_openai_to_account_vector(self, openai_vector):
        # if self.avg_post_to_account_model is None or openai_vector is None: return None # 古いモデル変数
        if self.dcor_filtered_avg_to_account_model is None or openai_vector is None: return None # 新しいモデル変数
        if openai_vector.shape[0] != self.openai_dim: self.stdout.write(self.style.ERROR(f"Input vector dim mismatch.")); return None
        try:
            with torch.no_grad():
                tensor = torch.tensor(openai_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                # output = self.avg_post_to_account_model(tensor) # 古いモデル呼び出し
                output = self.dcor_filtered_avg_to_account_model(tensor) # 新しいモデル呼び出し
                return output.squeeze(0).cpu().numpy()
        # except Exception as e: self.stdout.write(self.style.ERROR(f"Error applying AvgPostToAccountModel: {e}")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error applying DcorFilteredAvgPostToAccountModel: {e}")); return None

    def find_most_similar_bot(self, target_account_vector, bot_user_objects):
        if target_account_vector is None or not bot_user_objects: return None
        if target_account_vector.shape[0] != self.node2vec_dim:
             self.stdout.write(self.style.ERROR(f"Target vector dim ({target_account_vector.shape[0]}) != NODE2VEC_DIM ({self.node2vec_dim}). Model output error?"))
             return None
        bot_usernames = [user.username for user in bot_user_objects]
        bot_embeddings = UserEmbedding.objects.filter(user__username__in=bot_usernames, node2vec_vector__isnull=False).select_related('user')
        if not bot_embeddings.exists(): self.stdout.write(self.style.WARNING("No Node2Vec embeddings for bot accounts.")); return None
        bot_node2vec_vectors = []; bot_embedding_users = []
        for be in bot_embeddings:
            try:
                vec = np.array(be.node2vec_vector)
                if vec.shape == (self.node2vec_dim,): bot_node2vec_vectors.append(vec); bot_embedding_users.append(be.user)
            except (TypeError, ValueError): pass
        if not bot_node2vec_vectors: self.stdout.write(self.style.WARNING("No valid Node2Vec vectors for bots.")); return None
        similarity_results = []
        try:
            target_vec_reshaped = target_account_vector.reshape(1, -1)
            if target_vec_reshaped.shape[1] != self.node2vec_dim: self.stdout.write(self.style.ERROR("Reshaped target vector dim incorrect.")); return []
            similarity_scores = cosine_similarity(target_vec_reshaped, np.array(bot_node2vec_vectors))[0]
            for i, user_obj in enumerate(bot_embedding_users):
                 similarity_results.append({'user': user_obj, 'similarity': similarity_scores[i]})
            return similarity_results
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error during similarity calculation: {e}")); return []

    def create_post_as_similar_bot(self, post_author_user, content_summary, original_link):
        cleaned_summary = re.sub('<[^<]+?>', '', content_summary).strip();
        if not cleaned_summary: return None
        post_content = f"{self.post_identifier_prefix}\n\"{cleaned_summary}\""
        if len(post_content) > self.post_max_length:
            available_length = self.post_max_length - len(f"{self.post_identifier_prefix}\n\"\"") - 3;
            if available_length < 10: self.stdout.write(self.style.WARNING(f"Cannot fit summary.")); return None
            truncated_summary = cleaned_summary[:available_length] + "..."; post_content = f"{self.post_identifier_prefix}\n\"{truncated_summary}\"" ;
        if len(post_content) > self.post_max_length: self.stdout.write(self.style.WARNING(f"Final post too long.")); return None
        try:
            if Post.objects.filter(content__icontains=original_link).exists():
                self.stdout.write(self.style.WARNING(f"Post link exists: {original_link}"))
                return None
            post = Post.objects.create(user=post_author_user, content=post_content);
            self.stdout.write(self.style.SUCCESS(f"Created post ID {post.id} as {post_author_user.username}")); return post
        except Exception as e: self.stdout.write(self.style.ERROR(f"Failed post as {post_author_user.username}: {e}")); return None


    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Starting bot posting process (Using DcorFilteredAvgPostToAccountModel)..."))
        if not self.openai_available: self.stdout.write(self.style.ERROR("OpenAI key not available.")); return
        if not self.dcor_filtered_avg_to_account_model: self.stdout.write(self.style.ERROR("DcorFilteredAvgPostToAccountModel not loaded.")); return
        if not self.bot_users: self.stdout.write(self.style.ERROR("No valid bot accounts available.")); return

        unique_bot_usernames = list(set(BOT_ACCOUNT_USERNAMES)); required_bot_count = len(unique_bot_usernames)
        try:
            valid_vector_count = UserEmbedding.objects.filter(user__username__in=unique_bot_usernames,node2vec_vector__isnull=False).count()
            self.stdout.write(f"Checking embeddings for {required_bot_count} unique bot accounts...")
            self.stdout.write(f"Found {valid_vector_count} bot accounts with valid Node2Vec embeddings.")
            if valid_vector_count < required_bot_count:
                users_with_vectors = UserEmbedding.objects.filter(user__username__in=unique_bot_usernames,node2vec_vector__isnull=False).values_list('user__username', flat=True)
                missing_bots = [name for name in unique_bot_usernames if name not in users_with_vectors]
                self.stdout.write(self.style.ERROR(f"Missing Node2Vec embeddings for {required_bot_count - valid_vector_count} bots. Cannot proceed. Missing: {', '.join(missing_bots[:5])}{'...' if len(missing_bots) > 5 else ''}"))
                self.stdout.write(self.style.NOTICE("Hint: Run 'aggregate_user_vectors' command."))
                return
            else:
                 self.stdout.write(self.style.SUCCESS("All required bot accounts have Node2Vec embeddings."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error checking bot embeddings: {e}"))
            return

        if not RSS_FEED_URL_LIST: self.stdout.write(self.style.ERROR("RSS_FEED_URL_LIST empty.")); return
        selected_rss_url = random.choice(RSS_FEED_URL_LIST); self.stdout.write(f"Selected RSS feed: {selected_rss_url}")
        content_items = self.fetch_rss_summaries(selected_rss_url)
        if not content_items: self.stdout.write(self.style.WARNING(f"No content items from {selected_rss_url}.")); return

        processed_count = 0; posted_count = 0
        items_to_process = content_items[:self.process_item_limit]
        self.stdout.write(f"Attempting to process up to {len(items_to_process)} items...")

        for item in items_to_process:
            link = item.get('link'); title = item.get('title'); summary = item.get('summary')
            if not link or not title or not summary: continue
            processed_count += 1
            self.stdout.write(f"\n[{processed_count}/{len(items_to_process)}] Processing: {title}")
            cleaned_summary = re.sub('<[^<]+?>', '', summary).strip()
            if len(cleaned_summary) == 0: self.stdout.write(self.style.WARNING(f"Skipping '{title}': Empty summary.")); continue

            openai_embedding = self.get_openai_embedding(cleaned_summary)
            if openai_embedding is None: time.sleep(1); continue

            mapped_account_vector = self.map_openai_to_account_vector(openai_embedding)
            if mapped_account_vector is None:
                 self.stdout.write(self.style.WARNING(f"Could not map embedding for: {title}"))
                 time.sleep(1); continue

            similarity_results = self.find_most_similar_bot(mapped_account_vector, self.bot_users)
            if similarity_results is None or not similarity_results:
                 self.stdout.write(self.style.WARNING(f"Could not calculate similarities for: {title}"))
                 time.sleep(1); continue

            sorted_similarities = sorted(similarity_results, key=lambda x: x['similarity'], reverse=True)
            similar_bot_user = sorted_similarities[0]['user']
            self.stdout.write(f"--- Similarity scores for '{title[:50]}...' ---")
            for result in sorted_similarities[:10]: self.stdout.write(f"  {result['user'].username}: {result['similarity']:.4f}")
            if len(sorted_similarities) > 10: self.stdout.write("  ...")

            created_post = self.create_post_as_similar_bot(similar_bot_user, summary, link)
            if created_post: posted_count += 1; self.stdout.write("Waiting..."); time.sleep(3)
            else: time.sleep(0.5)

        self.stdout.write(self.style.SUCCESS(f"\nFinished. Processed {processed_count}, created {posted_count} posts."))
