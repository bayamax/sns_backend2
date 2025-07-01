from rest_framework.views import APIView
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
import os

from django.contrib.auth import get_user_model

from .serializers import UserSerializer
from .utils import verify_apple_identity_token

User = get_user_model()


class AppleLoginJWT(APIView):
    """Apple Sign-In 用カスタムログインビュー。

    iOS から送信される id_token (Apple の Identity Token) を検証し、
    ユーザーを作成／取得して SimpleJWT のアクセストークンを返す。
    レスポンスフォーマットはフロントエンド（SwiftUI 側）の期待に合わせ、

        {
            "token": "<access jwt>",
            "user": { ...UserSerializer fields... }
        }

    とする。
    """

    # Apple のコールバックではまだ認証されていないので AllowAny
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        # iOS 側実装では "id_token" で送信するが、保険として "identity_token" も受け付ける
        identity_token = request.data.get("id_token") or request.data.get("identity_token")
        username = request.data.get("username")  # 任意

        if not identity_token:
            return Response({"detail": "`id_token` is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            claims = verify_apple_identity_token(
                identity_token,
                audience=os.environ.get("APPLE_CLIENT_ID"),
            )
        except Exception as e:
            return Response({"detail": "Invalid identity token", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        apple_sub = claims.get("sub")
        email = claims.get("email")
        if not apple_sub:
            return Response({"detail": "Invalid Apple token"}, status=status.HTTP_400_BAD_REQUEST)

        # Apple の sub は一意のIDなのでユーザー名に含めておく
        default_username = f"apple_{apple_sub}"
        user, _ = User.objects.get_or_create(
            username=username or default_username,
            defaults={"email": email or ""},
        )

        refresh = RefreshToken.for_user(user)
        access_token = refresh.access_token

        user_data = UserSerializer(user, context={"request": request}).data

        return Response({
            "token": str(access_token),
            "user": user_data,
        }, status=status.HTTP_200_OK) 