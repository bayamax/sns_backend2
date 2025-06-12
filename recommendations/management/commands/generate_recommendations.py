import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.conf import settings
from posts.models import Post
from recommendations.models import UserEmbedding, UserRecommendation

User = get_user_model()

NODE2VEC_DIM = 128
OPENAI_DIM = 3072
MAX_POSTS = 50

class AttentionPooling(nn.Module):
def **init**(self, post_dim=OPENAI_DIM, acc_dim=NODE2VEC_DIM, n_head=8):
super().**init**()
self.mha = nn.MultiheadAttention(post_dim, n_head, batch_first=True, dropout=0.1)
self.ln  = nn.LayerNorm(post_dim)
self.proj = nn.Sequential(
nn.Linear(post_dim, acc_dim*2),
nn.GELU(),
nn.Linear(acc_dim*2, acc_dim)
)
self.sc  = nn.Linear(post_dim, 1, bias=False)

```
def forward(self, x, mask=None):
    x = F.normalize(x, p=2, dim=-1)
    att, _ = self.mha(x, x, x, key_padding_mask=(mask==0) if mask is not None else None)
    x = self.ln(x + att)
    scr = self.sc(x).squeeze(-1)
    if mask is not None:
        scr = scr.masked_fill(mask==0, -1e9)
    w = F.softmax(scr, dim=-1)
    acc = torch.sum(x * w.unsqueeze(-1), dim=1)
    return F.normalize(self.proj(acc), p=2, dim=-1), w
```

class FollowPredictor(nn.Module):
def **init**(self, acc_dim=NODE2VEC_DIM, hid=256):
super().**init**()
self.fe = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
self.te = nn.Sequential(nn.Linear(acc_dim, hid), nn.GELU())
self.head = nn.Sequential(
nn.Linear(hid*3, hid), nn.GELU(),
nn.Linear(hid, hid//2), nn.GELU(),
nn.Linear(hid//2, 1), nn.Sigmoid()
)

```
def forward(self, f, t):
    f, t = self.fe(f), self.te(t)
    return self.head(torch.cat([f, t, f*t], -1)).squeeze(-1)
```

class EndToEndFollowModel(nn.Module):
def **init**(self, post_dim=OPENAI_DIM, account_dim=NODE2VEC_DIM, hidden_dim=256):
super().**init**()
self.attention_pooling = AttentionPooling(post_dim, account_dim)
self.follow_predictor = FollowPredictor(account_dim, hidden_dim)

```
def forward(self, follower_posts, followee_posts, follower_masks=None, followee_masks=None):
    follower_vector, follower_attn = self.attention_pooling(follower_posts, follower_masks)
    followee_vector, followee_attn = self.attention_pooling(followee_posts, followee_masks)
    follow_prob = self.follow_predictor(follower_vector, followee_vector)
    return follow_prob, follower_vector, followee_vector, (follower_attn, followee_attn)
```

class Command(BaseCommand):
help = ‘ログ積み版：attention_pooling_follow_model.ptを使って投稿ベクトル→アカウントベクトル生成→フォロー推薦まで全てワンストップで処理します。’

```
def add_arguments(self, parser):
    parser.add_argument('--force', action='store_true', help='既存のUserEmbeddingを全削除して再生成')
    parser.add_argument('--top_k', type=int, default=10, help='リコメンド上位件数')

def log(self, message, level="INFO"):
    """ログ出力ヘルパー"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{level}] {message}"
    print(formatted_message)
    self.stdout.write(formatted_message)

def get_memory_usage(self):
    """メモリ使用量を取得"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"
    except ImportError:
        return "psutil未インストール"
    except Exception as e:
        return f"取得エラー: {e}"

def check_torch_health(self, device):
    """PyTorchの動作確認"""
    try:
        self.log("PyTorch動作確認テスト実行中...")
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.mm(test_tensor, test_tensor.T)
        self.log(f"PyTorch動作確認: OK (結果形状: {result.shape})")
        return True
    except Exception as e:
        self.log(f"PyTorch動作確認: 失敗 - {e}", "ERROR")
        return False

def validate_model_structure(self, model):
    """モデル構造の確認"""
    try:
        self.log("モデル構造の検証中...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"総パラメータ数: {total_params:,}")
        self.log(f"学習可能パラメータ数: {trainable_params:,}")
        
        if hasattr(model, 'attention_pooling'):
            self.log("✓ attention_pooling コンポーネント確認")
        if hasattr(model, 'follow_predictor'):
            self.log("✓ follow_predictor コンポーネント確認")
        
        return True
    except Exception as e:
        self.log(f"モデル構造検証エラー: {e}", "ERROR")
        return False

def handle(self, *args, **options):
    start_time = time.time()
    force = options.get('force', False)
    top_k = options.get('top_k', 10)
    
    self.log("=" * 80)
    self.log("🚀 ログ積み版推薦生成コマンド開始")
    self.log(f"📊 オプション: force={force}, top_k={top_k}")
    self.log(f"💾 初期メモリ使用量: {self.get_memory_usage()}")
    self.log("=" * 80)

    try:
        # 1. 環境情報の確認
        self.log("🔍 STEP 1: 環境情報の確認開始")
        self.log(f"📁 BASE_DIR: {settings.BASE_DIR}")
        self.log(f"📁 実行ディレクトリ: {os.getcwd()}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.log(f"🖥️  使用デバイス: {device}")
        
        if torch.cuda.is_available():
            self.log(f"🔥 CUDA デバイス数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.log(f"🔥 CUDA デバイス {i}: {torch.cuda.get_device_name(i)}")
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.log(f"🔥 CUDA メモリ総量: {memory_total:.1f}GB")
        
        if not self.check_torch_health(device):
            self.log("❌ PyTorch動作確認に失敗しました", "ERROR")
            return
        
        # 2. モデルファイルの確認
        self.log("🔍 STEP 2: モデルファイルの確認開始")
        model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'pretrained', 'attention_pooling_follow_model.pt')
        self.log(f"📄 モデルパス: {model_path}")
        
        if not os.path.exists(model_path):
            self.log(f"❌ モデルファイルが存在しません: {model_path}", "ERROR")
            parent_dir = os.path.dirname(model_path)
            if os.path.exists(parent_dir):
                files = os.listdir(parent_dir)
                self.log(f"📂 {parent_dir} 内のファイル: {files}")
            return
        
        file_size = os.path.getsize(model_path)
        self.log(f"📏 モデルファイルサイズ: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        with open(model_path, 'rb') as f:
            header = f.read(100)
            self.log(f"📄 ファイルヘッダー (最初50バイト): {header[:50]}")
            
            try:
                header_str = header.decode('utf-8')
                if 'version https://git-lfs.github.com' in header_str:
                    self.log("⚠️  Git LFSポインターファイルが検出されました！", "WARNING")
                    self.log("💡 'git lfs pull' を実行してください", "WARNING")
                    return
            except UnicodeDecodeError:
                self.log("✅ バイナリファイルを確認（正常なモデルファイル）")

        # 3. データベース情報の確認
        self.log("🔍 STEP 3: データベース情報の確認開始")
        user_count = User.objects.filter(is_staff=False, is_superuser=False).count()
        total_user_count = User.objects.count()
        post_count = Post.objects.count()
        embedding_count = UserEmbedding.objects.count()
        recommendation_count = UserRecommendation.objects.count()
        
        self.log(f"👥 一般ユーザー数: {user_count}")
        self.log(f"👥 全ユーザー数: {total_user_count}")
        self.log(f"📝 投稿数: {post_count}")
        self.log(f"🧠 既存UserEmbedding数: {embedding_count}")
        self.log(f"⭐ 既存推薦数: {recommendation_count}")

        posts_with_embedding = Post.objects.filter(embedding__isnull=False).count()
        posts_without_embedding = post_count - posts_with_embedding
        self.log(f"🧠 エンベディング付き投稿数: {posts_with_embedding}")
        self.log(f"❌ エンベディング無し投稿数: {posts_without_embedding}")
        
        if posts_with_embedding == 0:
            self.log("⚠️  投稿にエンベディングがありません！先にOpenAI埋め込みを実行してください", "WARNING")

        # 4. モデルの読み込み
        self.log("🔍 STEP 4: モデルの読み込み開始")
        model_load_start = time.time()
        
        try:
            self.log("🏗️  モデルオブジェクト作成中...")
            model = EndToEndFollowModel()
            self.log("✅ モデルオブジェクト作成完了")
            
            if not self.validate_model_structure(model):
                return
            
            self.log("📥 チェックポイント読み込み中...")
            ckpt = torch.load(model_path, map_location=device)
            self.log(f"✅ チェックポイント読み込み完了: {type(ckpt)}")
            
            if isinstance(ckpt, dict):
                self.log(f"🔑 チェックポイントキー数: {len(ckpt)}")
                self.log(f"🔑 チェックポイントキー例: {list(ckpt.keys())[:5]}")
                
                for i, (k, v) in enumerate(list(ckpt.items())[:3]):
                    if hasattr(v, 'shape'):
                        self.log(f"🔑 {k}: {v.shape}")
                    if i >= 2:
                        break
            
            self.log("🔄 キー名変換処理開始...")
            new_ckpt = {}
            conversion_count = 0
            
            for k, v in ckpt.items():
                if k.startswith('ap.'):
                    new_key = k.replace('ap.', 'attention_pooling.')
                    new_ckpt[new_key] = v
                    self.log(f"🔄 キー変換: {k} -> {new_key}")
                    conversion_count += 1
                elif k.startswith('fp.'):
                    new_key = k.replace('fp.', 'follow_predictor.')
                    new_ckpt[new_key] = v
                    self.log(f"🔄 キー変換: {k} -> {new_key}")
                    conversion_count += 1
                else:
                    new_ckpt[k] = v
            
            self.log(f"✅ キー変換完了: {conversion_count}個のキーを変換")
            self.log(f"📊 変換後のキー数: {len(new_ckpt)}")
            
            self.log("📥 モデルにstate_dict読み込み中...")
            model.load_state_dict(new_ckpt)
            self.log("✅ state_dict読み込み完了")
            
            self.log(f"🖥️  モデルをデバイス {device} に転送中...")
            model.to(device)
            model.eval()
            self.log("✅ モデル設定完了（評価モードに切り替え）")
            
            model_load_time = time.time() - model_load_start
            self.log(f"🎉 モデル読み込み全体完了！ 処理時間: {model_load_time:.2f}秒")
            self.log(f"💾 モデル読み込み後メモリ使用量: {self.get_memory_usage()}")
            
        except Exception as e:
            self.log(f"❌ モデル読み込みエラー: {e}", "ERROR")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"📋 詳細エラー情報:", "ERROR")
            for line in error_details.split('\n'):
                if line.strip():
                    self.log(f"   {line}", "ERROR")
            return

        # 5. UserEmbeddingの再生成
        self.log("🔍 STEP 5: UserEmbedding処理開始")
        if force:
            self.log("🧹 UserEmbedding強制再生成モード")
            deleted_count = UserEmbedding.objects.count()
            UserEmbedding.objects.all().delete()
            self.log(f"🗑️  既存UserEmbedding {deleted_count}件を削除しました")
        else:
            self.log("🔄 UserEmbedding増分更新モード（既存は保持）")
        
        embedding_start_time = time.time()
        self.generate_user_embeddings(device, model)
        embedding_time = time.time() - embedding_start_time
        self.log(f"⏱️  UserEmbedding処理時間: {embedding_time:.2f}秒")

        # 6. フォロー推薦計算
        self.log("🔍 STEP 6: フォロー推薦計算開始")
        recommendation_start_time = time.time()
        self.generate_recommendations(device, model, top_k)
        recommendation_time = time.time() - recommendation_start_time
        self.log(f"⏱️  推薦計算処理時間: {recommendation_time:.2f}秒")

        final_embedding_count = UserEmbedding.objects.count()
        final_recommendation_count = UserRecommendation.objects.count()
        
        total_time = time.time() - start_time
        self.log("=" * 80)
        self.log("🎉 全処理完了！")
        self.log(f"⏱️  総実行時間: {total_time:.2f}秒")
        self.log(f"🧠 最終UserEmbedding数: {final_embedding_count}")
        self.log(f"⭐ 最終推薦数: {final_recommendation_count}")
        self.log(f"💾 最終メモリ使用量: {self.get_memory_usage()}")
        self.log("=" * 80)

    except KeyboardInterrupt:
        self.log("⏸️  ユーザーによる中断", "WARNING")
    except Exception as e:
        self.log(f"💥 予期しないエラー: {e}", "ERROR")
        import traceback
        error_details = traceback.format_exc()
        self.log("📋 詳細エラー情報:", "ERROR")
        for line in error_details.split('\n'):
            if line.strip():
                self.log(f"   {line}", "ERROR")

def pad_posts(self, posts, max_length=MAX_POSTS):
    """投稿ベクトルをパディング"""
    if len(posts) == 0:
        return np.zeros((max_length, OPENAI_DIM), dtype=np.float32), np.zeros(max_length)
    
    if len(posts) >= max_length:
        return np.array(posts[:max_length]), np.ones(max_length)
    else:
        padded = np.zeros((max_length, OPENAI_DIM), dtype=np.float32)
        mask = np.zeros(max_length)
        padded[:len(posts)] = np.array(posts)
        mask[:len(posts)] = 1
        return padded, mask

def generate_user_embeddings(self, device, model):
    """投稿ベクトルからアカウントベクトルを生成"""
    self.log("🧠 アカウントベクトル生成処理開始")
    
    users = User.objects.filter(is_staff=False, is_superuser=False)
    total_users = users.count()
    self.log(f"👥 処理対象ユーザー数: {total_users}")
    
    if total_users == 0:
        self.log("⚠️  処理対象ユーザーが存在しません", "WARNING")
        return
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, user in enumerate(users):
        try:
            if i % 10 == 0 and i > 0:
                progress = (i / total_users) * 100
                self.log(f"📊 進捗: {i}/{total_users} ({progress:.1f}%) - 処理済み: {processed_count}, スキップ: {skipped_count}, エラー: {error_count}")
            
            posts = Post.objects.filter(user=user)
            post_count = posts.count()
            
            if post_count == 0:
                if i < 5:
                    self.log(f"👤 ユーザー {user.username} (ID:{user.id}): 投稿なし")
                skipped_count += 1
                continue
            
            post_vectors = []
            valid_posts = 0
            invalid_posts = 0
            
            for post in posts:
                if (hasattr(post, 'embedding') and post.embedding and 
                    post.embedding.vector and isinstance(post.embedding.vector, list) and 
                    len(post.embedding.vector) == OPENAI_DIM):
                    
                    vector = np.array(post.embedding.vector, dtype=np.float32)
                    vector_norm = np.linalg.norm(vector)
                    if vector_norm > 0:
                        normalized_vector = vector / vector_norm
                        post_vectors.append(normalized_vector)
                        valid_posts += 1
                    else:
                        invalid_posts += 1
                else:
                    invalid_posts += 1
            
            if not post_vectors:
                if i < 5:
                    self.log(f"👤 ユーザー {user.username} (ID:{user.id}): 投稿{post_count}件あるが、有効なベクトル0件")
                skipped_count += 1
                continue
            
            if i < 5:
                self.log(f"👤 ユーザー {user.username} (ID:{user.id}): 投稿{post_count}件中、有効{valid_posts}件、無効{invalid_posts}件")
            
            padded_posts, mask = self.pad_posts(post_vectors)
            
            with torch.no_grad():
                posts_tensor = torch.tensor(padded_posts, dtype=torch.float32).unsqueeze(0).to(device)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                
                account_vector, attention_weights = model.attention_pooling(posts_tensor, mask_tensor)
                account_vector_cpu = account_vector.squeeze(0).cpu().numpy()
                
                vector_norm = np.linalg.norm(account_vector_cpu)
                vector_mean = np.mean(account_vector_cpu)
                
                if i < 3:
                    self.log(f"📊 ベクトル統計 - norm: {vector_norm:.4f}, mean: {vector_mean:.4f}")
                
                account_vector_list = account_vector_cpu.tolist()
            
            user_embedding, created = UserEmbedding.objects.update_or_create(
                user=user,
                defaults={'node2vec_vector': account_vector_list}
            )
            
            action = "新規作成" if created else "更新"
            if i < 5:
                self.log(f"✅ ユーザー {user.username}: アカウントベクトル{action}完了")
            
            processed_count += 1
            
            if i % 50 == 0 and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            error_count += 1
            self.log(f"❌ ユーザー {user.username} (ID:{user.id}) 処理エラー: {e}", "ERROR")
            if error_count <= 3:
                import traceback
                self.log(f"📋 エラー詳細: {traceback.format_exc()}", "ERROR")
            continue
    
    self.log(f"🎯 アカウントベクトル生成完了")
    self.log(f"📊 結果: 処理成功={processed_count}, スキップ={skipped_count}, エラー={error_count}")
    self.log(f"💾 処理後メモリ使用量: {self.get_memory_usage()}")

def generate_recommendations(self, device, model, top_k):
    """アカウントベクトル間のフォロー推薦計算"""
    self.log("⭐ フォロー推薦計算処理開始")
    
    embeddings = UserEmbedding.objects.exclude(node2vec_vector__isnull=True)
    embedding_count = embeddings.count()
    self.log(f"🧠 アカウントベクトル保有ユーザー数: {embedding_count}")
    
    if embedding_count == 0:
        self.log("⚠️  アカウントベクトルが存在しません", "WARNING")
        return
    
    user_vectors = {}
    vector_load_errors = 0
    
    for embedding in embeddings:
        try:
            vector = np.array(embedding.node2vec_vector, dtype=np.float32)
            if vector.shape[0] != NODE2VEC_DIM:
                self.log(f"⚠️  ユーザー {embedding.user.id}: ベクトル次元不正 {vector.shape}", "WARNING")
                continue
            user_vectors[str(embedding.user.id)] = vector
        except Exception as e:
            vector_load_errors += 1
            if vector_load_errors <= 3:
                self.log(f"❌ ユーザー {embedding.user.id} のベクトル読み込みエラー: {e}", "ERROR")
    
    users = list(user_vectors.keys())
    valid_user_count = len(users)
    self.log(f"✅ 推薦計算対象ユーザー数: {valid_user_count}")
    self.log(f"❌ ベクトル読み込みエラー数: {vector_load_errors}")
    
    if valid_user_count == 0:
        self.log("❌ 有効なユーザーベクトルが存在しません", "ERROR")
        return

    total_recommendations = 0
    error_users = 0
    
    for i, user_id in enumerate(users):
        try:
            if i % 10 == 0 and i > 0:
                progress = (i / valid_user_count) * 100
                self.log(f"📊 推薦計算進捗: {i}/{valid_user_count} ({progress:.1f}%) - 生成済み推薦: {total_recommendations}")
            
            candidates = [uid for uid in users if uid != user_id]
            if not candidates:
                self.log(f"⚠️  ユーザー {user_id}: 推薦候補が存在しません", "WARNING")
                continue
                
            user_vec = user_vectors[user_id]
            results = []
            
            batch_size = 100
            
            for j in range(0, len(candidates), batch_size):
                batch_candidates = candidates[j:j + batch_size]
                
                user_batch = torch.tensor([user_vec] * len(batch_candidates), dtype=torch.float32, device=device)
                candidate_batch = torch.tensor([user_vectors[c_id] for c_id in batch_candidates], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    probs = model.follow_predictor(user_batch, candidate_batch)
                    probs_cpu = probs.cpu().numpy()
                    
                    for k, c_id in enumerate(batch_candidates):
                        results.append((c_id, float(probs_cpu[k])))
            
            results.sort(key=lambda x: x[1], reverse=True)
            top_results = results[:top_k]
            
            deleted_count = UserRecommendation.objects.filter(user_id=user_id).count()
            UserRecommendation.objects.filter(user_id=user_id).delete()
            
            created_recommendations = []
            for c_id, prob in top_results:
                recommendation = UserRecommendation.objects.create(
                    user_id=user_id,
                    recommended_user_id=c_id,
                    score=prob,
                    follow_probability=round(min(100.0, prob * 100), 1),
                    uncertainty=0.0
                )
                created_recommendations.append(recommendation)
            
            if i < 5:
                self.log(f"👤 ユーザー {user_id}: {deleted_count}件削除, {len(created_recommendations)}件新規作成")
                if created_recommendations:
                    top_score = created_recommendations[0].score
                    self.log(f"   最高スコア: {top_score:.4f}")
            
            total_recommendations += len(created_recommendations)
            
        except Exception as e:
            error_users += 1
            self.log(f"❌ ユーザー {user_id} の推薦計算エラー: {e}", "ERROR")
            if error_users <= 3:
                import traceback
                self.log(f"📋 エラー詳細: {traceback.format_exc()}", "ERROR")
            continue
    
    self.log(f"🎯 フォロー推薦計算完了")
    self.log(f"📊 結果: 総推薦数={total_recommendations}, エラーユーザー数={error_users}")
    self.log(f"💾 処理後メモリ使用量: {self.get_memory_usage()}")
```