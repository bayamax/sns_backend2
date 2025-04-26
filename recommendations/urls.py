# recommendations/urls.py

from django.urls import path
from .views import (
    RecommendationsView, RecommendationUpdateView, 
    VectorGenerationView
)

urlpatterns = [
    path('', RecommendationsView.as_view(), name='recommendations'),
    path('update', RecommendationUpdateView.as_view(), name='recommendation-update'),
    path('ai/generate-vector/<int:post_id>', VectorGenerationView.as_view(), name='vector-generation'),
]