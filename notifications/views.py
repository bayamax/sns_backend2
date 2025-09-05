from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Notification
from .models import Device
from .serializers import NotificationSerializer
from .serializers import DeviceSerializer

class NotificationListView(generics.ListAPIView):
    """通知一覧を取得するAPIビュー"""
    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Notification.objects.filter(recipient=self.request.user)


class UnreadNotificationListView(generics.ListAPIView):
    """未読通知を取得するAPIビュー"""
    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Notification.objects.filter(recipient=self.request.user, read=False)


class MarkNotificationAsReadView(APIView):
    """通知を既読にするAPIビュー"""
    permission_classes = [IsAuthenticated]
    
    def patch(self, request, pk):
        try:
            notification = Notification.objects.get(id=pk, recipient=request.user)
            notification.read = True
            notification.save()
            
            serializer = NotificationSerializer(notification)
            return Response(serializer.data)
        except Notification.DoesNotExist:
            return Response(
                {"detail": "通知が見つかりません"}, 
                status=status.HTTP_404_NOT_FOUND
            )


class MarkAllNotificationsAsReadView(APIView):
    """すべての通知を既読にするAPIビュー"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        notifications = Notification.objects.filter(recipient=request.user, read=False)
        count = notifications.count()
        
        # 一括更新
        notifications.update(read=True)
        
        return Response(
            {"detail": f"{count}件の通知を既読にしました"}, 
            status=status.HTTP_200_OK
        )


# ------------------------------
# Device 登録 / 削除 API
# ------------------------------


class DeviceRegisterView(APIView):
    """デバイストークンを登録・更新するAPIビュー"""

    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = DeviceSerializer(data=request.data)
        if serializer.is_valid():
            token = serializer.validated_data["token"]
            platform = serializer.validated_data.get("platform", Device.IOS)

            device, _ = Device.objects.update_or_create(
                token=token,
                defaults={
                    "user": request.user,
                    "platform": platform,
                },
            )

            return Response(DeviceSerializer(device).data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        token = request.data.get("token")
        if not token:
            return Response({"detail": "token は必須です"}, status=status.HTTP_400_BAD_REQUEST)

        Device.objects.filter(token=token, user=request.user).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)