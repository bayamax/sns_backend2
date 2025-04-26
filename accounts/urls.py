# accounts/urls.py

from django.urls import path
# simplejwtのビューをインポート
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from .views import (
    RegisterView, LoginView, UserProfileView, 
    FollowView, NotificationView, FollowStatusView,
    FollowersListView, FollowingListView, AccountDeleteView,
    BlockUserView, BlockedUsersListView
)

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    # simplejwt のトークン取得・リフレッシュ用エンドポイントを追加
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('users/profile/', UserProfileView.as_view(), name='user-profile'),
    path('users/<int:pk>/', UserProfileView.as_view(), name='user-detail'),
    path('users/<int:user_id>/follow/', FollowView.as_view(), name='user-follow'),
    path('users/<int:user_id>/follow-status/', FollowStatusView.as_view(), name='follow-status'),
    path('users/<int:user_id>/followers/', FollowersListView.as_view(), name='user-followers'),
    path('users/<int:user_id>/following/', FollowingListView.as_view(), name='user-following'),
    path('notifications/', NotificationView.as_view(), name='notifications'),
    path('me/delete/', AccountDeleteView.as_view(), name='account-delete'),
    path('<int:pk>/block/', BlockUserView.as_view(), name='block-user'),
    path('me/blocked/', BlockedUsersListView.as_view(), name='blocked-list'),
]