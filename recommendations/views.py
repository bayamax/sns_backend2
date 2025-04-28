# recommendations/views.py

from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import UserRecommendation, UserEmbedding
from .serializers import UserRecommendationSerializer
from accounts.models import Follow, Block
from django.db.models import Q
from django.contrib.auth import get_user_model

User = get_user_model()

class RecommendationsView(APIView):
    """ユーザー推薦ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        try:
            # ユーザーへの推薦を取得
            recommendations = UserRecommendation.objects.filter(user=request.user)
            
            # すでにフォローしているユーザーを除外
            following_users = Follow.objects.filter(follower=request.user).values_list('following_id', flat=True)
            recommendations = recommendations.exclude(recommended_user__in=following_users)
            
            # ★★★ ブロック関連ユーザーを除外 ★★★
            # 自分がブロックしているユーザーIDリストを取得
            blocked_user_ids = Block.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)
            # 自分をブロックしているユーザーIDリストを取得
            blocked_by_user_ids = Block.objects.filter(blocked=request.user).values_list('blocker_id', flat=True)
            # 除外対象のユーザーIDリスト (Setで重複排除)
            exclude_user_ids = set(blocked_user_ids) | set(blocked_by_user_ids)
            recommendations = recommendations.exclude(recommended_user_id__in=exclude_user_ids)
            # ★★★ ここまで追加 ★★★
            
            # 自分自身を除外
            recommendations = recommendations.exclude(recommended_user=request.user)
            
            # 上位10件を取得
            recommendations = recommendations.order_by('-score')[:10]
            
            # 常に配列を返す（存在チェックをせず、空でも配列としてシリアライズ）
            serializer = UserRecommendationSerializer(recommendations, many=True, context={'request': request})
            return Response(serializer.data)
        except Exception as e:
            # エラーログを詳細に出力
            import traceback
            print(f"推薦取得エラー: {str(e)}")
            print(traceback.format_exc())
            
            # エラー時も空の配列を返す
            return Response([])
    
class RecommendationUpdateView(APIView):
    """推薦更新ビュー（APIではなく管理コマンドで実行することを想定）"""
    permission_classes = [permissions.IsAdminUser]
    
    def post(self, request):
        # このエンドポイントは管理コマンドで実行される想定
        # 実際の処理は別途管理コマンドで実装する
        return Response({'detail': 'Recommendation update initiated'})

class VectorGenerationView(APIView):
    """ベクトル生成ビュー（APIではなく管理コマンドで実行することを想定）"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, post_id):
        # このエンドポイントは管理コマンドで実行される想定
        # 実際の処理は別途管理コマンドで実装する
        return Response({'detail': 'Vector generation initiated'})

# 以下は実際のAPIエンドポイントとしては使用せず、管理コマンドとして実装する想定
# class GenerateEmbeddingCommand(BaseCommand):
#     help = '新規投稿のOpenAI埋め込みベクトルを生成'
#     
#     def handle(self, *args, **options):
#         # 実装省略
#         pass
# 
# class AggregateVectorsCommand(BaseCommand):
#     help = 'ユーザーの投稿ベクトルを集約'
#     
#     def handle(self, *args, **options):
#         # 実装省略
#         pass
# 
# class ConvertEmbeddingsCommand(BaseCommand):
#     help = 'OpenAI埋め込みをNode2Vecベクトルに変換'
#     
#     def handle(self, *args, **options):
#         # 実装省略
#         pass
# 
# class PredictFolloweesCommand(BaseCommand):
#     help = 'フォロイー予測'
#     
#     def handle(self, *args, **options):
#         # 実装省略
#         pass