from django.urls import path
from . import views

app_name = 'notifications'

urlpatterns = [
    path('', views.NotificationListView.as_view(), name='list'),
    path('unread/', views.UnreadNotificationListView.as_view(), name='unread'),
    path('<int:pk>/read/', views.MarkNotificationAsReadView.as_view(), name='mark_read'),
    path('mark-all-read/', views.MarkAllNotificationsAsReadView.as_view(), name='mark_all_read'),
]