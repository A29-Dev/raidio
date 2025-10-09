from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("api/recent/", views.recent, name="recent"),
    path("api/wiki/", views.wiki_summary, name="wiki_summary"),
    path("wiki/<str:lang>/<path:title>/", views.wiki_proxy, name="wiki_proxy"),
]
