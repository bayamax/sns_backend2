from rest_framework.views import APIView
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
import os
from jose import jwt  # for debug pre-check
import logging

from django.contrib.auth import get_user_model

from .serializers import UserSerializer
from .utils import verify_apple_identity_token

User = get_user_model()
logger = logging.getLogger(__name__)


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

        # iOS ネイティブの Sign-in では aud がバンドルID、Web フローでは Service ID になる
        allowed_audiences = list(filter(None, [
            os.environ.get("APPLE_CLIENT_ID"),   # Service ID (Web フロー)
            os.environ.get("APPLE_BUNDLE_ID"),  # iOS アプリの Bundle ID (ネイティブ)
        ]))

        # ---- Debug: トークンの aud を検証前に表示 ----
        try:
            unverified_claims = jwt.get_unverified_claims(identity_token)
            logger.error("Apple login pre-check --------------")
            logger.error("token aud = %s", unverified_claims.get("aud"))
            logger.error("allowed_audiences = %s", allowed_audiences)
            logger.error("-------------------------------------")
        except Exception as dbg_e:
            logger.error("[Debug] unable to decode unverified claims: %s", dbg_e)

        # python-jose は audience に list を渡すとエラーになるため
        # 要素が 1 つのときだけ文字列で渡し、複数ある場合は None にして後で手動検証する
        if len(allowed_audiences) == 1:
            audience_param = allowed_audiences[0]
        else:
            audience_param = None  # jose 側では検証しない

        try:
            claims = verify_apple_identity_token(
                identity_token,
                audience=audience_param,
            )

            # ---- Debug logs: remove after verification ----
            logger.error("Apple login debug --------------")
            logger.error("claims[aud] = %s", claims.get("aud"))
            logger.error("allowed_audiences = %s", allowed_audiences)
            logger.error("---------------------------------")

        except Exception as e:
            return Response({"detail": "Invalid identity token", "error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # audience を手動検証（複数想定）
        if allowed_audiences and claims.get("aud") not in allowed_audiences:
            return Response({"detail": "Invalid audience", "error": f"aud={claims.get('aud')}"}, status=status.HTTP_400_BAD_REQUEST)

        apple_sub = claims.get("sub")
        email = claims.get("email")
        if not apple_sub:
            return Response({"detail": "Invalid Apple token"}, status=status.HTTP_400_BAD_REQUEST)

        # ---------------- ユーザー取得・作成 ----------------
        # 1) apple_sub で既存ユーザーを検索
        user = User.objects.filter(apple_sub=apple_sub).first()

        # 自動採番: user1, user2 ...
        def generate_username():
            last = User.objects.order_by("-id").first()
            next_num = (last.id + 1) if last else 1
            return f"user{next_num}"

        # 2) 見つからなければ新規作成
        if user is None:
            # フロントが明示 username を送ってきた場合はそれを使うが、重複時は採番
            base_username = username or generate_username()
            new_username = base_username
            while User.objects.filter(username=new_username).exists():
                new_username = generate_username()

            user = User(
                username=new_username,
                email=email or "",
                is_apple_only=True,
                apple_sub=apple_sub,
            )
            user.set_unusable_password()
            user.save()
        else:
            # 既存ユーザーが email 未設定で今回取得できたら入れる
            if email and not user.email:
                user.email = email
                user.save(update_fields=["email"])

        refresh = RefreshToken.for_user(user)
        access_token = refresh.access_token

        user_data = UserSerializer(user, context={"request": request}).data

        return Response({
            "access": str(access_token),
            "refresh": str(refresh),
            "user": user_data,
        }, status=status.HTTP_200_OK) 