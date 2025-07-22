# posts/urls.py

from django.urls import path
from .views import (
    TimelineView, PostView, PostDetailView, 
    PostLikeView, PostCommentsView, UserPostsView, ChildCommentsView,
    TestView, ReportPostView, GlobalTimelineView
)

urlpatterns = [
    path('timeline/', TimelineView.as_view(), name='timeline'),
    path('global/', GlobalTimelineView.as_view(), name='global-timeline'),
    path('users/', UserPostsView.as_view(), name='user-posts'),
    path('test/', TestView.as_view(), name='test-view'),
    path('user/<int:user_id>/', UserPostsView.as_view(), name='user-posts-by-id'),
    path('<int:pk>/like/', PostLikeView.as_view(), name='post-like'),
    path('<int:pk>/comments/', PostCommentsView.as_view(), name='post-comments'),
    path('<int:pk>/child-comments/', ChildCommentsView.as_view(), name='child-comments'),
    path('<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('', PostView.as_view(), name='post-create'),
    path('<int:pk>/report/', ReportPostView.as_view(), name='report-post'),
]