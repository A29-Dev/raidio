from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from .models import Chunk, Entity


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