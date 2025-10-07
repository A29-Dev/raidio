import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from .models import Chunk, Entity
from django.views.decorators.http import require_GET
from django.utils.text import slugify


@require_GET
def wiki_summary(request):
    """Proxy to Wikipedia Summary API."""
    title = request.GET.get("title", "").strip()
    if not title:
        return JsonResponse({"ok": False, "error": "missing title"}, status=400)

    # Basic normalization (Wikipedia likes Title_Case)
    norm = "_".join(w.capitalize() for w in title.split())
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{norm}"

    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 404:
            return JsonResponse({"ok": False, "error": "not found"}, status=404)
        r.raise_for_status()
        data = r.json()
        return JsonResponse({
            "ok": True,
            "title": data.get("title"),
            "extract": data.get("extract"),
            "thumbnail": (data.get("thumbnail") or {}).get("source"),
            "content_urls": (data.get("content_urls") or {}).get("desktop", {}),
        })
    except requests.RequestException as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=502)


def home(request):
    # You can pass config here (e.g., WS URL) if you want
    return render(request, 'dashboard/index.html', {
        'ws_url': 'ws://127.0.0.1:8000/live',
        'page_title': 'Raidio Dashboard'
    })

def recent(request):
    """Return the most recent chunks with entities (for initial page load)."""
    limit = int(request.GET.get("limit", 50))
    qs = (Chunk.objects
          .select_related("station")
          .prefetch_related("entities")
          .order_by("-t0")[:limit])
    items = []
    for c in qs:
        items.append({
            "station_id": c.station_id,
            "chunk_id": c.id,
            "t0": c.t0.isoformat(),
            "t1": c.t1.isoformat(),
            "text": c.text or "",
            "entities": [
                {
                    "type": e.typ,
                    "text": e.surface,
                    "lat": e.lat, "lon": e.lon,
                    "conf": e.conf,
                } for e in c.entities.all()
            ],
        })
    return JsonResponse({"items": items, "server_time": timezone.now().isoformat()})