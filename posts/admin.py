# posts/admin.py

from django.contrib import admin
from .models import Post, Like, Report

class PostAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'content_preview', 'likes_count', 'replies_count', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'content')
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    
    content_preview.short_description = 'Content'
    
    def replies_count(self, obj):
        return obj.post_replies.count()
    
    replies_count.short_description = 'Replies'

class LikeAdmin(admin.ModelAdmin):
    list_display = ('user', 'post', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'post__content')

class ReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'reported_post_link', 'reporter_link', 'reason', 'status', 'timestamp')
    list_filter = ('status', 'reason', 'timestamp')
    search_fields = ('reported_post__content', 'reporter__username', 'reported_post__user__username', 'detail')
    list_editable = ('status',)
    list_per_page = 50
    readonly_fields = ('reported_post_link', 'reporter_link', 'timestamp')

    def reported_post_link(self, obj):
        from django.urls import reverse
        from django.utils.html import format_html
        link = reverse("admin:posts_post_change", args=[obj.reported_post.id])
        return format_html('<a href="{}">{}</a>', link, obj.reported_post)
    reported_post_link.short_description = 'Reported Post'
    reported_post_link.admin_order_field = 'reported_post'

    def reporter_link(self, obj):
        from django.urls import reverse
        from django.utils.html import format_html
        link = reverse("admin:accounts_user_change", args=[obj.reporter.id])
        return format_html('<a href="{}">{}</a>', link, obj.reporter.username)
    reporter_link.short_description = 'Reporter'
    reporter_link.admin_order_field = 'reporter'

admin.site.register(Post, PostAdmin)
admin.site.register(Like, LikeAdmin)
admin.site.register(Report, ReportAdmin)