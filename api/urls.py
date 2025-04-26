# api/urls.py
from django.urls import path, include
from django.http import JsonResponse

def api_root(request):
    return JsonResponse({
        "message": "Welcome to SNS Backend API",
        "status": "online",
        "endpoints": {
            "accounts": "/api/accounts/",
            "posts": "/api/posts/",
            "recommendations": "/api/recommendations/",
            "notifications": "/api/notifications/"
        }
    })

urlpatterns = [
    path('', api_root, name='api-root'),
    path('accounts/', include('accounts.urls')),
    path('auth/', include('accounts.urls')),
    path('posts/', include('posts.urls')),
    path('recommendations/', include('recommendations.urls')),
    path('notifications/', include('notifications.urls')),
]
