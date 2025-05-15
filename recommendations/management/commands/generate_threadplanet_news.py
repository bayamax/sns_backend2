import time
import openai
import random
import numpy as np
import torch
import torch.nn as nn # DcorFilteredAvgPostToAccountModel のために追加
import os # os モジュールをインポート
from sklearn.metrics.pairwise import cosine_similarity
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from posts.models import Post
from recommendations.models import UserEmbedding
from recommendations.ml_models import ProbabilisticFollowPredictor # ProbabilisticFollowPredictor をインポート
# from pytrends.request import TrendReq

User = get_user_model()

GLOBAL_NEWS_BOT_USERNAME = "newsbot_Threadplanet"
NEWS_BOT_PREFIX = "newsbot_"
POST_MAX_LENGTH = 280
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# --- モデルパス (settings.py や環境変数から取得するのが望ましいが、ここでは直接記述) ---
# 実際のパスに置き換えてください
DCOR_MODEL_PATH = getattr(settings, 'RECOMMENDATION_MODEL_PATHS', {}).get('DCOR_AVG_TO_ACCOUNT_MODEL', 'recommendations/pretrained/dcor_filtered_avg_to_account_model.pt')
PROBA_FOLLOW_MODEL_PATH = getattr(settings, 'RECOMMENDATION_MODEL_PATHS', {}).get('PROBABILISTIC_FOLLOW_MODEL', 'recommendations/pretrained/probabilistic_followee_model.pt')

NODE2VEC_DIM = 128 # from generate_recommendations.py (UserEmbedding.node2vec_vectorの次元)
OPENAI_DIM_FOR_DCOR_MODEL = 3072 # DcorFilteredAvgPostToAccountModelの入力次元 (投稿ベクトル)
RECOMMENDATION_THRESHOLD = 0.5 # 関連スコアの仮の閾値

# --- DcorFilteredAvgPostToAccountModel の定義 (ユーザー提供コードまたは generate_user_node2vec.py から) ---
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
    def __init__(self, input_dim=OPENAI_DIM_FOR_DCOR_MODEL, output_dim=NODE2VEC_DIM, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(input_dim, 1024, dropout),
            ResidualBlock(1024, 512, dropout),
            ResidualBlock(512, 256, dropout),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.model(x)
# --- ここまでモデル定義 ---

class Command(BaseCommand):
    help = f'Generates news posts for {GLOBAL_NEWS_BOT_USERNAME} and user-specific news bots based on recommendation logic.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_available = False
        self.dcor_model = None
        self.proba_follow_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        #api_key_from_settings = getattr(settings, 'OPENAI_API_KEY', None)
        api_key_from_settings = ""
        if api_key_from_settings:
            openai.api_key = api_key_from_settings
            self.openai_available = True
        else:
            self.stdout.write(self.style.ERROR("OpenAI API key not found."))

        # モデルのロード
        self._load_dcor_model()
        self._load_proba_follow_model()

    def _load_dcor_model(self):
        try:
            self.dcor_model = DcorFilteredAvgPostToAccountModel(input_dim=OPENAI_DIM_FOR_DCOR_MODEL, output_dim=NODE2VEC_DIM)
            # settings.BASE_DIR を使って絶対パスを構築 (モデルパスが相対の場合)
            # from django.conf import settings as django_settings
            # import os
            # model_path = DCOR_MODEL_PATH 
            # if not os.path.isabs(model_path):
            #    model_path = os.path.join(django_settings.BASE_DIR, model_path)
            model_path = os.path.join(settings.BASE_DIR, DCOR_MODEL_PATH) if not os.path.isabs(DCOR_MODEL_PATH) else DCOR_MODEL_PATH

            self.dcor_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.dcor_model.to(self.device)
            self.dcor_model.eval()
            self.stdout.write(self.style.SUCCESS(f"DcorFilteredAvgPostToAccountModel loaded from {model_path}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading DcorFilteredAvgPostToAccountModel: {e}"))
            self.dcor_model = None

    def _load_proba_follow_model(self):
        try:
            # ProbabilisticFollowPredictor の hidden_dim は generate_recommendations.py のものを参照
            # FOLLOWEE_HIDDEN_DIM = getattr(settings, 'RECOMMENDATION_SETTINGS', {}).get('FOLLOWEE_HIDDEN_DIM', 64)
            followee_hidden_dim = 64 # 仮の値。generate_recommendations.py と合わせる必要あり
            self.proba_follow_model = ProbabilisticFollowPredictor(NODE2VEC_DIM, followee_hidden_dim)
            # model_path = PROBA_FOLLOW_MODEL_PATH
            # if not os.path.isabs(model_path):
            #    model_path = os.path.join(settings.BASE_DIR, model_path)
            model_path = os.path.join(settings.BASE_DIR, PROBA_FOLLOW_MODEL_PATH) if not os.path.isabs(PROBA_FOLLOW_MODEL_PATH) else PROBA_FOLLOW_MODEL_PATH

            self.proba_follow_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.proba_follow_model.to(self.device)
            self.proba_follow_model.eval()
            self.stdout.write(self.style.SUCCESS(f"ProbabilisticFollowPredictor loaded from {model_path}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading ProbabilisticFollowPredictor: {e}"))
            self.proba_follow_model = None

    # --- Methods for Global Trend News (newsbot_Threadplanet) ---
    def get_global_news_bot_user(self):
        try:
            bot_user = User.objects.get(username=GLOBAL_NEWS_BOT_USERNAME)
            return bot_user
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Global news bot user '{GLOBAL_NEWS_BOT_USERNAME}' not found. Please create this user."))
            return None

    def get_trending_keywords(self):
        self.stdout.write(self.style.NOTICE("Fetching trending keywords (pytrends or dummy)..."))
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='ja-JP', tz=360)
            trending_searches_df = pytrends.trending_searches(pn='japan')
            if trending_searches_df is None or trending_searches_df.empty:
                self.stdout.write(self.style.WARNING("No trending keywords found from pytrends."))
                return []
            keywords = trending_searches_df[0].tolist()[:3]
            if not keywords:
                self.stdout.write(self.style.WARNING("Trending keywords list is empty after processing."))
                return []
            self.stdout.write(self.style.SUCCESS(f"Found keywords from pytrends: {', '.join(keywords)}"))
            return keywords
        except ImportError:
            self.stdout.write(self.style.ERROR("pytrends library is not installed. Falling back to dummy keywords."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error fetching trends from pytrends: {e}. Falling back to dummy keywords."))
        dummy_keywords = ["AIの最新動向", "再生可能エネルギー", "メタバースの今後"]
        self.stdout.write(self.style.SUCCESS(f"Using dummy keywords: {', '.join(dummy_keywords)}"))
        return dummy_keywords

    def generate_global_news_content(self, keyword):
        if not self.openai_available: return None
        self.stdout.write(f"Generating global news content for keyword: '{keyword}'...")
        try:
            prompt = f"""以下のキーワードに関する一般的な最新情報を元に、200文字程度の短いニュース記事を作成してください。スタイルは客観的かつ簡潔に。冒頭に「【トレンドヘッドライン】」と付けてください。キーワード: {keyword}記事:"""
            response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=250, temperature=0.7, n=1, stop=None)
            content = response.choices[0].text.strip()
            if len(content) > POST_MAX_LENGTH: content = content[:POST_MAX_LENGTH-3] + "..."
            self.stdout.write(self.style.SUCCESS(f"Generated global content for '{keyword}': {content[:30]}..."))
            return content
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error generating global content for '{keyword}': {e}")); return None

    def create_post_as_bot(self, bot_user, content, is_global_news=True):
        if not content or not bot_user:
            return
        if Post.objects.filter(user=bot_user, content=content).exists():
            self.stdout.write(self.style.WARNING(f"Skipping duplicate content for {bot_user.username}."))
            return
        try:
            post = Post.objects.create(user=bot_user, content=content)
            prefix = "Global" if is_global_news else "User-specific"
            self.stdout.write(self.style.SUCCESS(f"{prefix} post ID {post.id} created for {bot_user.username}."))
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error creating post for {bot_user.username}: {e}"))

    def handle_global_trend_news(self):
        self.stdout.write(self.style.NOTICE(f"--- Starting Global News Generation for {GLOBAL_NEWS_BOT_USERNAME} ---"))
        global_bot_user = self.get_global_news_bot_user()
        if not global_bot_user or not self.openai_available: self.stdout.write(self.style.WARNING("Global bot or OpenAI not ready.")); return
        keywords = self.get_trending_keywords()
        if not keywords: self.stdout.write(self.style.WARNING("No global keywords to process.")); return
        for keyword in keywords:
            generated_content = self.generate_global_news_content(keyword)
            if generated_content: self.create_post_as_bot(global_bot_user, generated_content, is_global_news=True); time.sleep(3)
            self.stdout.write("-" * 20)
        self.stdout.write(self.style.SUCCESS(f"--- Global News Generation for {GLOBAL_NEWS_BOT_USERNAME} completed. ---"))

    def get_openai_embedding(self, text):
        if not self.openai_available or not text: return None
        try:
            response = openai.Embedding.create(input=text, model=OPENAI_EMBEDDING_MODEL)
            return np.array(response['data'][0]['embedding'])
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error getting OpenAI embedding for '{text[:30]}...': {e}")); return None

    def get_post_as_account_vector(self, post_content):
        if not self.dcor_model or not post_content:
            return None
        openai_embedding = self.get_openai_embedding(post_content)
        if openai_embedding is None:
            return None
        try:
            with torch.no_grad():
                input_tensor = torch.tensor(openai_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
                account_vector = self.dcor_model(input_tensor).squeeze(0).cpu().numpy()
                return account_vector
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error converting post content to account vector: {e}"))
            return None

    def get_recommendation_score(self, user_vector, post_account_vector):
        """ ProbabilisticFollowPredictor を使ってスコアを計算 """
        if not self.proba_follow_model or user_vector is None or post_account_vector is None:
            return 0.0 # スコア計算不可
        try:
            with torch.no_grad():
                user_tensor = torch.tensor(user_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                post_tensor = torch.tensor(post_account_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                # ProbabilisticFollowPredictor は (user_A_vec, user_B_vec) を期待
                score = self.proba_follow_model(user_tensor, post_tensor).item()
                return score
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error getting recommendation score with ProbaFollowModel: {e}"))
            return 0.0

    def get_relevant_global_posts_for_user(self, target_user, global_bot_user, count=3):
        self.stdout.write(self.style.NOTICE(f"Getting relevant global posts for {target_user.username} using ProbaFollowModel..."))
        if not global_bot_user or not self.openai_available or not self.dcor_model or not self.proba_follow_model:
            self.stdout.write(self.style.WARNING("Required models or OpenAI not available for relevance calculation."))
            return []
        try:
            user_embedding_obj = UserEmbedding.objects.get(user=target_user)
            if not user_embedding_obj.node2vec_vector:
                self.stdout.write(self.style.WARNING(f"Node2Vec vector not found for user {target_user.username}.")); return []
            user_vector = np.array(user_embedding_obj.node2vec_vector) # .reshape(1,-1) はモデル内で想定
        except UserEmbedding.DoesNotExist:
            self.stdout.write(self.style.WARNING(f"UserEmbedding not found for user {target_user.username}.")); return []
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error fetching user vector for {target_user.username}: {e}")); return []

        global_posts = Post.objects.filter(user=global_bot_user).order_by('-created_at')[:20]
        if not global_posts.exists(): self.stdout.write(self.style.WARNING(f"No global posts by {global_bot_user.username}.")); return []

        self.stdout.write(self.style.NOTICE(f"Calculating recommendation scores for {global_posts.count()} global posts..."))
        post_scores = []
        for post in global_posts:
            post_account_vector = self.get_post_as_account_vector(post.content)
            if post_account_vector is not None:
                score = self.get_recommendation_score(user_vector, post_account_vector)
                post_scores.append((post, score))
            time.sleep(0.2) # API or heavy computation delay mitigation
        
        if not post_scores: self.stdout.write(self.style.WARNING("No scores could be calculated.")); return []
        post_scores.sort(key=lambda x: x[1], reverse=True)
        top_posts = [item[0] for item in post_scores if item[1] >= RECOMMENDATION_THRESHOLD][:count]
        
        self.stdout.write(self.style.SUCCESS(f"Selected {len(top_posts)} relevant global posts for {target_user.username} (Threshold: {RECOMMENDATION_THRESHOLD})."))
        return top_posts

    # --- Methods for User-Specific News (newsbot_<username>) ---
    def get_or_create_user_news_bot(self, target_username):
        bot_username = f"{NEWS_BOT_PREFIX}{target_username}"
        try:
            news_bot_user, created = User.objects.get_or_create(username=bot_username, defaults={'is_staff': False, 'is_active': True})
            if created: news_bot_user.set_unusable_password(); news_bot_user.save(); self.stdout.write(self.style.SUCCESS(f"Created news bot user: {bot_username}"))
            return news_bot_user
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error getting/creating news bot for {target_username}: {e}")); return None

    def generate_personalized_news_content(self, target_user, source_posts):
        if not self.openai_available or not source_posts: return None
        self.stdout.write(f"Generating personalized news for {target_user.username}...")
        source_contents = "\n".join([f"- {p.content[:100]}..." for p in source_posts])
        try:
            prompt = f"""ユーザー「{target_user.username}」さん向けの特別なニュースダイジェストです。以下の記事群は、{target_user.username}さんの関心に近い可能性のある最近のトレンドニュースです。これらの情報を踏まえ、特に{target_user.username}さんが興味を持ちそうな点を強調し、150文字程度で要約・編集してください。記事の冒頭には「【{target_user.username}さんへ今日の注目ニュース】」と付けてください。参考記事:{source_contents}ダイジェスト記事:"""
            response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt, max_tokens=200, temperature=0.75, n=1, stop=None)
            content = response.choices[0].text.strip()
            if len(content) > POST_MAX_LENGTH: content = content[:POST_MAX_LENGTH-3] + "..."
            self.stdout.write(self.style.SUCCESS(f"Generated personalized content for {target_user.username}: {content[:30]}..."))
            return content
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error generating personalized content for {target_user.username}: {e}")); return None

    # --- リコメンドリスト操作 --- 
    def add_recommendation(self, user, recommended_bot_user, score=100.0): # スコアは仮
        if not user or not recommended_bot_user: return
        from recommendations.models import UserRecommendation # ローカルインポート
        # 既存の同じおすすめを削除（あれば）
        UserRecommendation.objects.filter(user=user, recommended_user=recommended_bot_user).delete()
        # 新しいおすすめを追加
        UserRecommendation.objects.create(user=user, recommended_user=recommended_bot_user, score=score, follow_probability=score, uncertainty=0)
        self.stdout.write(self.style.SUCCESS(f"Added {recommended_bot_user.username} to {user.username}'s recommendations with score {score}."))

    def remove_recommendation(self, user, recommended_bot_user_to_remove):
        if not user or not recommended_bot_user_to_remove: return
        from recommendations.models import UserRecommendation # ローカルインポート
        deleted_count, _ = UserRecommendation.objects.filter(user=user, recommended_user=recommended_bot_user_to_remove).delete()
        if deleted_count > 0:
            self.stdout.write(self.style.SUCCESS(f"Removed {recommended_bot_user_to_remove.username} from {user.username}'s recommendations."))

    def handle_user_specific_news(self):
        self.stdout.write(self.style.NOTICE("--- Starting User-Specific News Generation ---"))
        if not self.openai_available or not self.dcor_model or not self.proba_follow_model:
            self.stdout.write(self.style.ERROR("Required models or OpenAI not available for user-specific news. Exiting."))
            return

        global_bot_user = self.get_global_news_bot_user()
        if not global_bot_user: self.stdout.write(self.style.WARNING("Global news bot not found.")); return

        users_to_process = User.objects.filter(userembedding__node2vec_vector__isnull=False, is_staff=False, is_superuser=False).exclude(username__startswith=NEWS_BOT_PREFIX).distinct()[:2] # Botとadmin以外、2名に制限
        
        if not users_to_process.exists(): self.stdout.write(self.style.WARNING("No target users found.")); return
        self.stdout.write(self.style.NOTICE(f"Found {users_to_process.count()} users with account vectors to process."))

        for target_user in users_to_process:
            self.stdout.write(f"Processing user: {target_user.username}")
            user_news_bot = self.get_or_create_user_news_bot(target_user.username)
            if not user_news_bot: continue

            # メインニュースアカウントの推薦を削除
            self.remove_recommendation(target_user, global_bot_user)

            relevant_posts = self.get_relevant_global_posts_for_user(target_user, global_bot_user, count=2)
            if not relevant_posts:
                self.stdout.write(self.style.WARNING(f"No relevant global posts for {target_user.username}."))
                # 個別ニュースがなくても、個別ニュースBot自体は推薦リストに追加しておく（空でもフォローできるように）
                self.add_recommendation(target_user, user_news_bot, score=90.0) # スコアは調整
                continue

            personalized_content = self.generate_personalized_news_content(target_user, relevant_posts)
            if personalized_content:
                self.create_post_as_bot(user_news_bot, personalized_content, is_global_news=False)
                time.sleep(3)
            
            # 個別ニュースBotをリコメンドに追加
            self.add_recommendation(target_user, user_news_bot, score=100.0) # 記事があれば高スコア
            self.stdout.write("*" * 20)
        self.stdout.write(self.style.SUCCESS("--- User-Specific News Generation completed. ---"))

    def handle_global_news_recommendations(self):
        self.stdout.write(self.style.NOTICE("--- Updating Recommendations for Users without Vectors ---"))
        global_bot_user = self.get_global_news_bot_user()
        if not global_bot_user: self.stdout.write(self.style.WARNING("Global news bot not found.")); return

        # アカウントベクトル未割り当てのユーザー（Botとadmin除く）
        users_without_vectors = User.objects.filter(embedding__node2vec_vector__isnull=True, is_staff=False, is_superuser=False).exclude(username__startswith=NEWS_BOT_PREFIX)
        if not users_without_vectors.exists(): self.stdout.write(self.style.NOTICE("No users without vectors to update recommendations for.")); return
        
        self.stdout.write(self.style.NOTICE(f"Found {users_without_vectors.count()} users without vectors."))
        for user in users_without_vectors:
            # 既存のメインニュース推薦を削除（念のため）
            self.remove_recommendation(user, global_bot_user)
            # メインニュースアカウントを推薦リストに追加
            self.add_recommendation(user, global_bot_user, score=110.0) # 高めのスコアで強制追加

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("=== Starting Full News Generation & Recommendation Cycle ==="))
        # 1. グローバルニュースの生成
        self.handle_global_trend_news()
        self.stdout.write("\n" + "="*40 + "\n")
        # 2. グローバルニュースの推薦（ベクトル未所持ユーザー向け）
        self.handle_global_news_recommendations()
        self.stdout.write("\n" + "="*40 + "\n")
        # 3. ユーザー個別ニュースの生成と推薦（ベクトル所持ユーザー向け）
        self.handle_user_specific_news()
        self.stdout.write(self.style.SUCCESS("=== Full News Generation & Recommendation Cycle completed. ===")) 