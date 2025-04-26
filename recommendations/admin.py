# recommendations/admin.py

from django.contrib import admin
from .models import UserRecommendation, PostEmbedding, UserEmbedding

class UserRecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'recommended_user', 'score', 'follow_probability', 'uncertainty', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'recommended_user__username')

class PostEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('post', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('post__user__username', 'post__content')

class UserEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('user', 'updated_at')
    list_filter = ('updated_at',)
    search_fields = ('user__username',)

admin.site.register(UserRecommendation, UserRecommendationAdmin)
admin.site.register(PostEmbedding, PostEmbeddingAdmin)
admin.site.register(UserEmbedding, UserEmbeddingAdmin)