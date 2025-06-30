# accounts/views.py

import os
from rest_framework import status, permissions, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from .serializers import (
    UserSerializer,
    RegisterSerializer,
    LoginSerializer,
    UserProfileSerializer,
    UserListSerializer,
    AppleLoginSerializer,
)
from .utils import verify_apple_identity_token
from notifications.serializers import NotificationSerializer
from .models import Follow, User, Block
from notifications.models import Notification
from django.contrib.auth import get_user_model
from django.shortcuts import get_object_or_404

User = get_user_model()

class RegisterView(APIView):
    """ユーザー登録ビュー"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            
            # レスポンスデータの作成
            response_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'token': str(refresh.access_token)
            }
            
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    """ログインビュー"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            
            refresh = RefreshToken.for_user(user)
            
            # UserSerializer を使ってユーザー情報を取得 (snake_case になる)
            user_data = UserSerializer(user, context={'request': request}).data
            
            # トークンを追加
            response_data = user_data
            # キーを 'password' から 'token' に変更
            response_data['token'] = str(refresh.access_token)
            
            return Response(response_data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AppleLoginView(APIView):
    """Apple IDを使用したログインビュー"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = AppleLoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        identity_token = serializer.validated_data["identity_token"]
        username = serializer.validated_data.get("username")

        try:
            claims = verify_apple_identity_token(
                identity_token, audience=os.environ.get("APPLE_CLIENT_ID")
            )
        except Exception as e:
            return Response({"detail": "Invalid identity token", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        apple_sub = claims.get("sub")
        email = claims.get("email")

        if not apple_sub:
            return Response({"detail": "Invalid Apple token"}, status=status.HTTP_400_BAD_REQUEST)

        user, created = User.objects.get_or_create(
            username=username or f"apple_{apple_sub}",
            defaults={"email": email or ""},
        )

        refresh = RefreshToken.for_user(user)
        user_data = UserSerializer(user, context={"request": request}).data
        user_data["token"] = str(refresh.access_token)
        return Response(user_data)

class UserProfileView(APIView):
    """ユーザープロフィールビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk=None):
        user = request.user if pk is None else User.objects.filter(pk=pk).first()
        
        if not user:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = UserSerializer(user, context={'request': request})
        return Response(serializer.data)
    
    def put(self, request):
        # デバッグ情報の出力
        print("プロフィール更新リクエスト受信:")
        print(f"Content-Type: {request.content_type}")
        print(f"データ: {request.data}")
        print(f"ファイル: {request.FILES}")
    
        try:
            # multipart/form-data形式のリクエストを処理
            if 'profile_image' in request.FILES:
                # プロフィール画像のアップロード処理
                profile_image = request.FILES['profile_image']
                request.user.profile_image = profile_image
                print(f"画像名: {profile_image.name}, サイズ: {profile_image.size}バイト")
            
                # 他のフィールドの更新
                if 'username' in request.data:
                    request.user.username = request.data['username']
                if 'bio' in request.data:
                    request.user.bio = request.data['bio']
                
                request.user.save()
            
                print(f"プロフィール画像を更新しました: {request.user.profile_image.name}")
                serializer = UserSerializer(request.user, context={'request': request})
                return Response(serializer.data)
            else:
                # 通常のJSON形式のリクエスト処理
                serializer = UserSerializer(request.user, data=request.data, partial=True, context={'request': request})
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data)
            
                print(f"バリデーションエラー: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            import traceback
            print(f"プロフィール更新エラー: {str(e)}")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FollowView(APIView):
    """フォロー/アンフォロービュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, user_id):
        try:
            user_to_follow = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # 自分自身をフォローすることはできない
        if request.user.id == user_id:
            return Response({'detail': "You can't follow yourself"}, status=status.HTTP_400_BAD_REQUEST)
        
        # すでにフォローしている場合は何もしない
        if Follow.objects.filter(follower=request.user, following=user_to_follow).exists():
            return Response({'detail': 'Already following'}, status=status.HTTP_200_OK)
        
        # フォロー関係を作成
        Follow.objects.create(follower=request.user, following=user_to_follow)
        
        # 通知の作成
        Notification.objects.create(
            recipient=user_to_follow,
            sender=request.user,
            notification_type=Notification.FOLLOW
        )
        
        return Response({'success': True}, status=status.HTTP_201_CREATED)
    
    def delete(self, request, user_id):
        try:
            user_to_unfollow = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # フォロー関係を削除
        follow = Follow.objects.filter(follower=request.user, following=user_to_unfollow).first()
        if follow:
            follow.delete()
            return Response({'success': True}, status=status.HTTP_200_OK)
        
        return Response({'detail': 'Not following'}, status=status.HTTP_400_BAD_REQUEST)

class NotificationView(APIView):
    """通知ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        notifications = Notification.objects.filter(recipient=request.user)
        serializer = NotificationSerializer(notifications, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        # すべての通知を既読にする
        Notification.objects.filter(recipient=request.user, read=False).update(read=True)
        return Response({'success': True})

class FollowStatusView(APIView):
    """フォロー状態確認ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, user_id):
        try:
            user_to_check = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # フォロー関係をチェック
        is_following = Follow.objects.filter(
            follower=request.user, 
            following=user_to_check
        ).exists()
        
        return Response({'is_following': is_following}, status=status.HTTP_200_OK)

class FollowersListView(APIView):
    """フォロワーリスト取得ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, user_id):
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # フォロワーを取得
        followers = User.objects.filter(following__following=user)
        
        # シリアライズして返す
        serializer = UserListSerializer(followers, many=True, context={'request': request})
        return Response(serializer.data)

class FollowingListView(APIView):
    """フォロー中リスト取得ビュー"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, user_id):
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({'detail': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # フォロー中ユーザーを取得
        following = User.objects.filter(followers__follower=user)
        
        # シリアライズして返す
        serializer = UserListSerializer(following, many=True, context={'request': request})
        return Response(serializer.data)

class AccountDeleteView(generics.GenericAPIView):
    """
    認証済みユーザー自身のアカウントを削除するビュー。
    """
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, *args, **kwargs):
        user = request.user
        try:
            # ユーザーを削除 (関連データは on_delete=CASCADE で削除されることを想定)
            user.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            # エラーハンドリング (ログ記録など)
            print(f"Account deletion failed for user {user.id}: {e}")
            return Response({"detail": "アカウントの削除中にエラーが発生しました。"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class BlockUserView(APIView):
    """ユーザーをブロックまたはアンブロックする"""
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk): # POSTでブロック
        user_to_block = get_object_or_404(User, pk=pk)
        if request.user == user_to_block:
            return Response({"detail": "自分自身をブロックすることはできません。"}, status=status.HTTP_400_BAD_REQUEST)
        
        # 既にブロック済みか確認
        if Block.objects.filter(blocker=request.user, blocked=user_to_block).exists():
             return Response({"detail": "既にブロック済みです。"}, status=status.HTTP_400_BAD_REQUEST)

        Block.objects.create(blocker=request.user, blocked=user_to_block)
        return Response({"detail": f"{user_to_block.username} をブロックしました。"}, status=status.HTTP_201_CREATED)

    def delete(self, request, pk): # DELETEでアンブロック
        user_to_unblock = get_object_or_404(User, pk=pk)
        block_instance = Block.objects.filter(blocker=request.user, blocked=user_to_unblock).first()
        if block_instance:
            block_instance.delete()
            return Response({"detail": f"{user_to_unblock.username} のブロックを解除しました。"}, status=status.HTTP_204_NO_CONTENT)
        else:
            return Response({"detail": "このユーザーはブロックしていません。"}, status=status.HTTP_404_NOT_FOUND)

class BlockedUsersListView(generics.ListAPIView):
    """ブロックしているユーザーのリストを取得"""
    permission_classes = [permissions.IsAuthenticated]
    # serializer_class = SimpleUserSerializer # 必要なら後で設定
    # ★注意: serializer_class 未設定だとエラーになる可能性あり。簡易的な UserSerializer を使うか、後で設定が必要。
    # とりあえず簡易的にユーザー名を返すようにします
    def list(self, request, *args, **kwargs):
        blocked_user_ids = Block.objects.filter(blocker=self.request.user).values_list('blocked_id', flat=True)
        blocked_users = User.objects.filter(id__in=blocked_user_ids)
        # 本当はSerializerを使うべき
        data = [{"id": user.id, "username": user.username} for user in blocked_users]
        return Response(data)

class UserDetailView(generics.RetrieveUpdateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        data = serializer.data

        return Response(data)
    