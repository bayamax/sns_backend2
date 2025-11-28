# recommendations/admin.py

from django.contrib import admin
from .models import UserRecommendation, PostEmbedding, UserEmbedding, Community, CommunityMembership

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

class CommunityAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'get_member_count', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name',)
    
    def get_member_count(self, obj):
        return obj.members.count()
    get_member_count.short_description = 'メンバー数'

class CommunityMembershipAdmin(admin.ModelAdmin):
    list_display = ('user', 'community', 'is_settled', 'assigned_at', 'settled_at')
    list_filter = ('is_settled', 'community', 'assigned_at')
    search_fields = ('user__username',)

admin.site.register(UserRecommendation, UserRecommendationAdmin)
admin.site.register(PostEmbedding, PostEmbeddingAdmin)
admin.site.register(UserEmbedding, UserEmbeddingAdmin)
admin.site.register(Community, CommunityAdmin)
admin.site.register(CommunityMembership, CommunityMembershipAdmin)