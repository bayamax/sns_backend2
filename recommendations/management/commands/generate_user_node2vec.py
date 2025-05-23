# recommendations/management/commands/generate_user_node2vec.py

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings # settingsをインポート
from django.db.models import Q
from sklearn.metrics.pairwise import cosine_similarity
import re # 正規表現をインポート

# posts と recommendations のモデルをインポート
from posts.models import Post
from recommendations.models import PostEmbedding, UserEmbedding

# BOTアカウントリストを定義
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

# ハードコードする設定値
HARDCODED_OPENAI_DIM = 3072
HARDCODED_NODE2VEC_DIM = 128
# HARDCODED_AVG_POST_TO_ACCOUNT_MODEL_PATH = 'recommendations/pretrained/avg_post_to_account_model.pt' # 古いパス
# HARDCODED_DCOR_MODEL_PATH = '/Users/oobayashikoushin/Desktop/test/sns_backend/recommendations/pretrained/dcor_filtered_avg_to_account_model.pt' # 古い絶対パス
HARDCODED_DCOR_MODEL_PATH = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'dcor_filtered_avg_to_account_model.pt') # settings.BASE_DIR を使ったパス

# モデル定義 (古いものは削除し、新しいものを追加)
# class AvgPostToAccountModel(nn.Module): ... (削除)

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
    # input_dim, output_dim は __init__ で渡される
    # dropout のデフォルト値はユーザー提供の学習コードからDROPOUT_RATE = 0.2 を想定
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

User = get_user_model()

class Command(BaseCommand):
    help = 'Generates user Node2Vec vectors from averaged post OpenAI embeddings using DcorFilteredAvgPostToAccountModel, excluding bots.' # help メッセージ更新

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration for users that already have Node2Vec embeddings'
        )
        parser.add_argument(
            '--user_id',
            type=int,
            help='Generate Node2Vec embedding for a specific user (by ID, excluding bots)'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # インスタンス変数として設定値を保持
        self.openai_dim = HARDCODED_OPENAI_DIM
        self.node2vec_dim = HARDCODED_NODE2VEC_DIM
        # self.avg_post_to_account_model_path = HARDCODED_AVG_POST_TO_ACCOUNT_MODEL_PATH # 古いパス変数
        self.dcor_model_path = HARDCODED_DCOR_MODEL_PATH # 新しいパス変数
        # モデルロードは handle 内で行うか、ここでロードするか選択可能
        # ここでロードする場合は get_device も __init__ で呼ぶ
        self.device = self.get_device()
        # self.avg_post_to_account_model = self.load_avg_post_to_account_model() # 古いモデルロード呼び出し
        self.dcor_filtered_avg_to_account_model = self.load_dcor_filtered_avg_to_account_model() # 新しいモデルロード呼び出し

    def get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try: torch.zeros(1, device="mps"); return torch.device("mps")
            except Exception: pass
        return torch.device("cpu")

    # def load_avg_post_to_account_model(self): ... (古いメソッドを削除)
    def load_dcor_filtered_avg_to_account_model(self):
        # model = AvgPostToAccountModel(input_dim=self.openai_dim, output_dim=self.node2vec_dim) # 古いクラス
        model = DcorFilteredAvgPostToAccountModel(input_dim=self.openai_dim, output_dim=self.node2vec_dim) # 新しいクラス
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
        except RuntimeError as e:
             self.stdout.write(self.style.ERROR(f"Error loading state_dict: {e}"))
             # self.stdout.write(self.style.WARNING("Ensure model definition matches saved file."))
             self.stdout.write(self.style.WARNING("Ensure DcorFilteredAvgPostToAccountModel definition matches saved file."))
             return None
        # except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading AvgPostToAccountModel: {e}")); return None
        except Exception as e: self.stdout.write(self.style.ERROR(f"Error loading DcorFilteredAvgPostToAccountModel: {e}")); return None


    def handle(self, *args, **options):
        force = options.get('force', False)
        user_id = options.get('user_id')

        # if not self.avg_post_to_account_model: # 古いモデルチェック
        #      self.stdout.write(self.style.ERROR("AvgPostToAccountModel not loaded. Cannot generate Node2Vec vectors. Exiting."))
        #      return
        if not self.dcor_filtered_avg_to_account_model: # 新しいモデルチェック
             self.stdout.write(self.style.ERROR("DcorFilteredAvgPostToAccountModel not loaded. Cannot generate Node2Vec vectors. Exiting."))
             return

        # 対象ユーザーを取得 (ボットを除外)
        if user_id:
            users = User.objects.filter(id=user_id).exclude(username__in=BOT_ACCOUNT_USERNAMES)
            if not users.exists():
                user_check = User.objects.filter(id=user_id).first()
                if user_check and user_check.username in BOT_ACCOUNT_USERNAMES:
                     self.stdout.write(self.style.WARNING(f"User ID {user_id} ({user_check.username}) is a bot account. Skipping."))
                else:
                     self.stdout.write(self.style.ERROR(f"User with ID {user_id} not found or is excluded."))
                return
        else:
            users = User.objects.exclude(username__in=BOT_ACCOUNT_USERNAMES)

        user_count = users.count()
        self.stdout.write(f"Processing {user_count} users (excluding bots)")

        # 各ユーザーについて処理
        for user in tqdm(users, desc="Generating Node2Vec vectors"):
            try:
                # 既存チェック (force=False の場合)
                if not force and UserEmbedding.objects.filter(user=user, node2vec_vector__isnull=False, openai_vector__isnull=False).exists():
                    continue

                # ユーザーの投稿を取得
                posts = Post.objects.filter(user=user)
                if not posts.exists():
                    continue

                # 投稿のOpenAIベクトルを取得
                post_vectors = []
                for post in posts:
                    try:
                        embedding = PostEmbedding.objects.get(post=post)
                        if embedding.vector and isinstance(embedding.vector, list) and len(embedding.vector) == self.openai_dim:
                            post_vectors.append(np.array(embedding.vector, dtype=np.float32))
                    except PostEmbedding.DoesNotExist:
                        pass

                if not post_vectors:
                    continue

                # 1. 平均ベクトルを計算
                avg_post_openai_vector = np.mean(post_vectors, axis=0)

                # 2. モデルで Node2Vec ベクトルを予測
                predicted_node2vec_vector = None
                try:
                    with torch.no_grad():
                        input_tensor = torch.tensor(avg_post_openai_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                        # output_tensor = self.avg_post_to_account_model(input_tensor) # 古いモデル呼び出し
                        output_tensor = self.dcor_filtered_avg_to_account_model(input_tensor) # 新しいモデル呼び出し
                        predicted_node2vec_vector = output_tensor.squeeze(0).cpu().numpy()
                except Exception as model_e:
                    self.stdout.write(self.style.ERROR(f"Error applying model for user {user.id}: {model_e}"))
                    continue

                if predicted_node2vec_vector is None or predicted_node2vec_vector.shape[0] != self.node2vec_dim:
                     self.stdout.write(self.style.WARNING(f"Invalid Node2Vec vector shape for user {user.id}"))
                     continue

                # ユーザー埋め込みを保存
                user_embedding, created = UserEmbedding.objects.update_or_create(
                    user=user,
                    defaults={
                        'openai_vector': avg_post_openai_vector.tolist(),      # 平均ベクトル
                        'node2vec_vector': predicted_node2vec_vector.tolist() # モデル出力
                    }
                )
                # if created: self.stdout.write(f"Created Node2Vec for {user.id}") # ログ調整

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing user {user.id}: {str(e)}"))

        self.stdout.write(self.style.SUCCESS("Node2Vec vector generation process completed.")) 