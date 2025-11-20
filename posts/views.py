# posts/views.py

from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Q, Count, Exists, OuterRef, Prefetch
from django.shortcuts import get_object_or_404
import logging

from .models import Post, Like, Report
from .serializers import PostSerializer, PostCreateSerializer, LikeSerializer
from accounts.models import Follow, User, Block
from notifications.models import Notification
from recommendations.models import UserRecommendation
from .news_bot import maybe_trigger_async

logger = logging.getLogger(__name__)

class TimelineView(APIView):
    """タイムラインビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        logger.info(f"TimelineView GET request received for user: {request.user.username} (ID: {request.user.id})")
        # ニュースボット: タイムライン閲覧時に確率で裏投稿（最小フック）
        maybe_trigger_async()

        try:
            # ブロック関係のユーザーIDを取得
            blocked_user_ids = list(Block.objects.filter(blocker=request.user).values_list('blocked_id', flat=True))
            logger.info(f"Blocked user IDs (by me): {blocked_user_ids}")

            blocked_by_user_ids = list(Block.objects.filter(blocked=request.user).values_list('blocker_id', flat=True))
            logger.info(f"User IDs blocking me: {blocked_by_user_ids}")

            blocked_related_user_ids = set(blocked_user_ids) | set(blocked_by_user_ids)
            logger.info(f"All blocked related user IDs: {blocked_related_user_ids}")

            # コミュニティメンバーの投稿を取得（自分を含む）
            community_user_ids = [request.user.id]  # 自分を含める
            try:
                from recommendations.models import CommunityMembership
                my_membership = request.user.community_membership
                community_member_ids = list(
                    CommunityMembership.objects.filter(community=my_membership.community)
                    .values_list('user_id', flat=True)
                )
                community_user_ids = community_member_ids
                logger.info(f"Community user IDs (including self): {community_user_ids}")
            except Exception as e:
                # コミュニティ未所属の場合は自分のみ
                logger.info(f"User not in community or error: {e}. Showing only own posts.")

            # コミュニティメンバー（自分含む）の投稿を取得
            posts_query = Post.objects.filter(
                user_id__in=community_user_ids,
                parent_post__isnull=True
            ).exclude(
                user_id__in=blocked_related_user_ids
            ).select_related('user').prefetch_related(
                Prefetch('likes', queryset=Like.objects.filter(user=request.user), to_attr='user_like')
            ).annotate(
                likesCount=Count('likes'),
                repliesCount=Count('post_replies')
            ).order_by('-created_at')

            posts_list = list(posts_query)
            logger.info(f"Number of community posts fetched: {len(posts_list)}")

            # isLiked フィールドを追加
            for post in posts_list:
                post.isLiked = hasattr(post, 'user_like') and len(post.user_like) > 0
                post.is_from_followed_user = True  # コミュニティ投稿として扱う
                logger.info(f"  Fetched Community Post ID: {post.id}, User ID: {post.user_id}")

            serializer = PostSerializer(posts_list, many=True, context={'request': request})
            logger.info(f"TimelineView successfully returning {len(serializer.data)} posts.")
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error in TimelineView GET for user {request.user.id}: {e}", exc_info=True)
            return Response({"detail": "タイムラインの取得中にエラーが発生しました。"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PostView(APIView):
    """投稿作成・取得ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        # ユーザーの全投稿を取得（親投稿のみ）
        posts = Post.objects.filter(
            user=request.user,
            parent_post__isnull=True
        ).select_related('user').order_by('-created_at')
        serializer = PostSerializer(posts, many=True, context={'request': request})
        return Response(serializer.data)
    
    def post(self, request):
        serializer = PostCreateSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            post = serializer.save()
            response_serializer = PostSerializer(post, context={'request': request})
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PostDetailView(APIView):
    """投稿詳細ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk):
        print(f"投稿詳細取得リクエスト: post_id={pk}, user={request.user.username}")
        try:
            post = Post.objects.select_related('user').get(pk=pk)
            print(f"投稿が見つかりました: post_id={pk}, content={post.content[:30]}...")
        except Post.DoesNotExist:
            print(f"投稿が見つかりません: post_id={pk}")
            return Response({'error': '投稿が見つかりません。'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = PostSerializer(post, context={'request': request})
        return Response(serializer.data)
    
    def put(self, request, pk):
        try:
            post = Post.objects.get(pk=pk, user=request.user)
        except Post.DoesNotExist:
            return Response({'detail': 'Post not found or you do not have permission'}, 
                           status=status.HTTP_404_NOT_FOUND)
        
        serializer = PostCreateSerializer(post, data=request.data)
        if serializer.is_valid():
            updated_post = serializer.save()
            response_serializer = PostSerializer(updated_post, context={'request': request})
            return Response(response_serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        try:
            post = Post.objects.get(pk=pk, user=request.user)
        except Post.DoesNotExist:
            return Response({'detail': 'Post not found or you do not have permission'}, 
                           status=status.HTTP_404_NOT_FOUND)
        
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class UserPostsView(APIView):
    """特定ユーザーの投稿一覧ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, user_id=None):
        try:
            # ブロックしているユーザーのIDを取得
            blocked_user_ids = list(Block.objects.filter(blocker=request.user).values_list('blocked_id', flat=True))
            logger.info(f"Blocked user IDs for posts view: {blocked_user_ids}")

            # クエリパラメータでIDの配列が指定されている場合
            if user_id is None and 'ids' in request.query_params:
                user_ids = request.query_params.get('ids', '').split(',')
                user_ids = [int(uid) for uid in user_ids if uid.isdigit()]
                
                if not user_ids:
                    return Response({'detail': 'No valid user IDs provided'}, 
                                   status=status.HTTP_400_BAD_REQUEST)
                
                # 複数ユーザーの投稿を取得
                posts = Post.objects.filter(
                    user_id__in=user_ids,
                    parent_post__isnull=True
                ).exclude(
                    user_id__in=blocked_user_ids
                ).select_related('user').order_by('-created_at')
                
                serializer = PostSerializer(posts, many=True, context={'request': request})
                return Response(serializer.data)
            
            # 単一のユーザーIDが指定されている場合
            elif user_id is not None:
                target_user = User.objects.get(pk=user_id)
                # 現在のユーザーがブロックしている場合は表示しない
                if target_user.id in blocked_user_ids:
                    return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
                
                # 親投稿のみを取得
                posts = Post.objects.filter(
                    user=target_user,
                    parent_post__isnull=True
                ).select_related('user').order_by('-created_at')
                
                serializer = PostSerializer(posts, many=True, context={'request': request})
                return Response(serializer.data)
            
            # どちらも指定されていない場合
            else:
                return Response({'detail': 'User ID or IDs required'}, 
                               status=status.HTTP_400_BAD_REQUEST)
                
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            import traceback
            print(f"ユーザー投稿取得エラー: {str(e)}")
            print(traceback.format_exc())
            return Response({'detail': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PostLikeView(APIView):
    """投稿いいねビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, pk):
        print(f"いいねリクエスト: post_id={pk}, user={request.user.username}")
        try:
            post = Post.objects.select_related('user').get(pk=pk)
            print(f"投稿が見つかりました: post_id={pk}, content={post.content[:30]}...")
        except Post.DoesNotExist:
            print(f"投稿が見つかりません: post_id={pk}")
            return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # すでにいいねしているかチェック
        is_liked = Like.objects.filter(user=request.user, post=post).exists()
        print(f"現在のいいね状態: is_liked={is_liked}")
        
        if is_liked:
            # いいねを削除
            Like.objects.filter(user=request.user, post=post).delete()
            liked = False
            print(f"いいねを削除: post_id={pk}, user={request.user.username}")
        else:
            # いいねを作成
            Like.objects.create(user=request.user, post=post)
            liked = True
            print(f"いいねを作成: post_id={pk}, user={request.user.username}")
            
            # 通知は signals (notifications.utils.post_liked_notification) で自動作成されるため
            # ここでは重複作成しない
        
        # 更新された投稿を返す
        updated_post = Post.objects.select_related('user').get(pk=pk)
        print(f"更新後のいいね数: {updated_post.likes_count}")
        serializer = PostSerializer(updated_post, context={'request': request})
        return Response({
            'liked': liked,
            'post': serializer.data
        })

class PostCommentsView(APIView):
    """投稿へのコメント（返信）ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk):
        try:
            post = Post.objects.get(pk=pk)
        except Post.DoesNotExist:
            return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # 直接の返信のみを取得（ネストされていないもの）
        replies = Post.objects.filter(
            parent_post=post
        ).select_related('user')
        
        # デバッグ情報をログに出力
        print(f"コメントを取得: 投稿ID {pk}, 件数 {replies.count()}")
        
        serializer = PostSerializer(replies, many=True, context={'request': request})
        return Response(serializer.data)
    
    def post(self, request, pk):
        try:
            parent_post = Post.objects.select_related('user').get(pk=pk)
        except Post.DoesNotExist:
            return Response({'detail': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # デバッグ情報
        print(f"コメント作成リクエスト: {request.data}")
        
        # リクエストデータのコピーを作成
        data = request.data.copy()
        # parent_postを設定
        data['parent_post'] = pk
        
        serializer = PostCreateSerializer(data=data, context={'request': request})
        if serializer.is_valid():
            # コメントを保存
            reply = serializer.save()
            
            # 通知は signals (notifications.utils.post_created_notification) で自動作成
            
            print(f"コメントを作成しました: ID {reply.id}")
            return Response(PostSerializer(reply, context={'request': request}).data, status=status.HTTP_201_CREATED)
        
        print(f"コメントバリデーションエラー: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ChildCommentsView(APIView):
    """子コメント取得ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk):
        try:
            parent_post = Post.objects.get(pk=pk)
        except Post.DoesNotExist:
            return Response({'detail': 'Parent post not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # 子コメントを取得
        child_comments = Post.objects.filter(parent_post=parent_post).select_related('user')
        serializer = PostSerializer(child_comments, many=True, context={'request': request})
        return Response(serializer.data)


class ReportPostView(APIView):
    """投稿を報告する"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        post_to_report = get_object_or_404(Post, pk=pk)

        # 既に報告済みか確認
        if Report.objects.filter(reporter=request.user, reported_post=post_to_report).exists():
            return Response({"detail": "既に報告済みです。"}, status=status.HTTP_400_BAD_REQUEST)

        # TODO: reason や detail をリクエストボディから受け取る処理を追加する場合はここに実装
        reason = request.data.get('reason', None)
        detail = request.data.get('detail', None)

        Report.objects.create(
            reporter=request.user,
            reported_post=post_to_report,
            reason=reason,
            detail=detail
        )
        return Response({"detail": "投稿を報告しました。"}, status=status.HTTP_201_CREATED)

# ===========================
# MapSNS 用: 全ユーザー投稿タイムライン
# ===========================
class GlobalTimelineView(APIView):
    """全ユーザーの親投稿を返すタイムライン（MapSNS 用・公開）"""
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        # 未ログインの場合 request.user は AnonymousUser となり id を持たないため "guest" として記録する
        logger.info(f"GlobalTimelineView GET request by user {getattr(request.user, 'id', 'guest')}")

        try:
            # ブロック関連のユーザーを除外 (未ログイン時は AnonymousUser なのでクエリをスキップ)
            if request.user.is_authenticated:
                blocked_user_ids = set(Block.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)) | \
                                   set(Block.objects.filter(blocked=request.user).values_list('blocker_id', flat=True))
            else:
                blocked_user_ids = set()

            # 匿名ユーザーの場合は like 判定を付けない
            if request.user.is_authenticated:
                like_qs = Prefetch('likes', queryset=Like.objects.filter(user=request.user), to_attr='user_like')
            else:
                like_qs = Prefetch('likes', queryset=Like.objects.none(), to_attr='user_like')

            posts_query = Post.objects.filter(parent_post__isnull=True)\
                                   .exclude(user_id__in=blocked_user_ids)\
                                   .select_related('user')\
                                   .prefetch_related(like_qs)\
                                   .annotate(likesCount=Count('likes'), repliesCount=Count('post_replies'))\
                                   .order_by('-created_at')

            posts_list = list(posts_query)
            for post in posts_list:
                post.isLiked = request.user.is_authenticated and hasattr(post, 'user_like') and len(post.user_like) > 0
                post.is_from_followed_user = False  # 全体タイムラインでは区別しない

            serializer = PostSerializer(posts_list, many=True, context={'request': request})
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error in GlobalTimelineView: {e}", exc_info=True)
            return Response({"detail": "タイムラインの取得中にエラーが発生しました。"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)