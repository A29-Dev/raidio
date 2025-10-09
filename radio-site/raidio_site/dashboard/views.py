import re
import requests
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.utils import timezone
from django.views.decorators.http import require_GET
from django.views.decorators.cache import cache_page
from django.utils.encoding import iri_to_uri

from .models import Chunk, Entity
from django.utils.text import slugify  # if needed elsewhere

# ---- Optional: HTML sanitization ----
try:
    import bleach
except ImportError:
    bleach = None  # recommend: pip install bleach

# ---- Bleach allowlists ----
ALLOWED_TAGS = [
    "a", "p", "b", "i", "em", "strong", "u", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6", "figure", "figcaption",
    "img", "table", "thead", "tbody", "tr", "th", "td",
    "code", "pre", "blockquote", "hr", "span", "div",
    "sup", "sub", "br"
]
ALLOWED_ATTRS = {
    "*": ["class", "id", "title", "role", "data-*", "aria-*"],
    "a": ["href", "title", "rel", "target"],
    "img": ["src", "srcset", "alt", "width", "height", "loading", "decoding"],
    "th": ["colspan", "rowspan", "scope"],
    "td": ["colspan", "rowspan"],
}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def _sanitize_html(html: str) -> str:
    """Sanitize incoming HTML; if bleach not installed, return as-is."""
    if not bleach:
        return html
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True
    )


def _normalize_title(raw: str) -> str:
    """Normalize Wikipedia title format."""
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 _\-()%./]", "", raw or "").strip()
    return cleaned.replace(" ", "_")


def _rewrite_urls(html: str, lang: str) -> str:
    """
    Wikipedia mobile-html includes relative URLs (/wiki/Foo, /w/...).
    Rewrite to absolute paths so images and links work.
    """
    base = f"https://{lang}.wikipedia.org"

    def abs_href(m):
        url = m.group(1)
        if url.startswith("/"):
            return f'href="{base}{url}"'
        return f'href="{url}"'

    def abs_src(m):
        url = m.group(1)
        if url.startswith("/"):
            return f'src="{base}{url}"'
        return f'src="{url}"'

    def abs_srcset(m):
        srcset = m.group(1)
        parts = []
        for chunk in srcset.split(","):
            c = chunk.strip()
            if not c:
                continue
            segs = c.split()
            if not segs:
                continue
            u = segs[0]
            if u.startswith("/"):
                u = base + u
            parts.append(" ".join([u, *segs[1:]]))
        return f'srcset="{", ".join(parts)}"'

    html = re.sub(r'href="([^"]+)"', abs_href, html, flags=re.IGNORECASE)
    html = re.sub(r'src="([^"]+)"', abs_src, html, flags=re.IGNORECASE)
    html = re.sub(r'srcset="([^"]+)"', abs_srcset, html, flags=re.IGNORECASE)
    return html


# ---- Wiki endpoints ----
@require_GET
def wiki_summary(request):
    """Proxy to Wikipedia Summary API (lightweight JSON extract)."""
    title = request.GET.get("title", "").strip()
    if not title:
        return JsonResponse({"ok": False, "error": "missing title"}, status=400)

    norm = _normalize_title(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{iri_to_uri(norm)}"

    try:
        r = requests.get(url, timeout=6)
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


@require_GET
@cache_page(60 * 60 * 12)  # 12 hours
def wiki_proxy(request, lang: str, title: str):
    """
    Server-side proxy that fetches sanitized, link-fixed mobile HTML
    for a Wikipedia page so it can render inside the RAIDIO dashboard.
    """
    lang = (lang or "en").lower().strip()
    title = _normalize_title(title)
    if not title:
        return HttpResponseBadRequest("Missing title")

    mobile_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/mobile-html/{iri_to_uri(title)}"
    headers = {
        "User-Agent": "RAIDIO/1.0 (wiki overlay)",
        "Accept": "text/html",
    }

    try:
        r = requests.get(mobile_url, headers=headers, timeout=10)
        if r.status_code == 404:
            # Graceful fallback to summary if page not found
            summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{iri_to_uri(title)}"
            sr = requests.get(summary_url, headers={"User-Agent": headers["User-Agent"]}, timeout=8)
            if sr.status_code == 404:
                return JsonResponse({"ok": False, "error": "not found"}, status=404)
            sr.raise_for_status()
            data = sr.json()
            html = f"<h2>{data.get('title','')}</h2><p>{data.get('extract','')}</p>"
            return JsonResponse({
                "ok": True,
                "title": data.get("title", title.replace('_', ' ')),
                "html": _sanitize_html(html)
            })

        r.raise_for_status()
        html = _rewrite_urls(r.text, lang)
        sanitized = _sanitize_html(html)
        return JsonResponse({
            "ok": True,
            "title": title.replace("_", " "),
            "html": sanitized
        })

    except requests.RequestException as e:
        return JsonResponse({"ok": False, "error": f"wiki lookup failed: {e}"}, status=502)


# ---- Dashboard endpoints ----
def home(request):
    return render(request, 'dashboard/index.html', {
        'ws_url': 'ws://127.0.0.1:8000/live',
        'page_title': 'Raidio Dashboard'
    })


def recent(request):
    """Return the most recent chunks with entities."""
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
                    "lat": e.lat,
                    "lon": e.lon,
                    "conf": e.conf,
                } for e in c.entities.all()
            ],
        })

    return JsonResponse({"items": items, "server_time": timezone.now().isoformat()})
