import os, sys, time, json, uuid, importlib, tempfile, subprocess, datetime, signal, re, math, wave, array
from collections import deque, defaultdict

# Runtime deps
import redis, psycopg
import requests
from urllib.parse import quote_plus
import spacy
from faster_whisper import WhisperModel

# UI deps (terminal VU + transcript)
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from colorama import Fore, Style, init as colorama_init


# ──────────────────────────────────────────────────────────────
# 0) GLOBAL CONFIG / CHANNELS
# ──────────────────────────────────────────────────────────────
EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "radio_events")

# Terminal / UI globals
colorama_init()
LOG_MAX = 80
TRANSCRIPT_LOG = deque(maxlen=LOG_MAX)

# ──────────────────────────────────────────────────────────────
# 1) VU METER UI HELPERS
# ──────────────────────────────────────────────────────────────
def render_ui(vu_bar_str: str, vu_hint: str, avg_rms: float | None):
    """Build the Rich layout: header VU + transcript panel (latest at bottom)."""
    header_text = Text(vu_bar_str or "", style="bold green")
    if vu_hint:
        header_text.append(f"\n{vu_hint}", style="yellow")
    if avg_rms is not None:
        header_text.append(f"\nAvg level: {avg_rms:0.1f} dBFS", style="cyan")

    header = Panel(header_text, title="🎚️ Live VU", border_style="green", padding=(1,2))

    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    for line in TRANSCRIPT_LOG:
        table.add_row(line)

    body = Panel(table, title="🎙️ Transcript", border_style="blue", padding=(1,2))

    layout = Layout()
    layout.split_column(
        Layout(header, size=6),   # pinned at top
        Layout(body),             # scrolls underneath
    )
    return layout


def wav_rms_peak_dbfs(wav_path):
    """Return (rms_dbfs, peak_dbfs) for a mono 16-bit PCM WAV."""
    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1, "expected mono"
        assert w.getsampwidth() == 2, "expected 16-bit"
        n = w.getnframes()
        frames = w.readframes(n)
    samples = array.array("h", frames)  # signed 16-bit
    if not samples:
        return -120.0, -120.0
    peak = max(abs(s) for s in samples)

    acc = 0
    for s in samples:
        acc += s * s
    rms = math.sqrt(acc / len(samples))

    def to_dbfs(x):
        if x <= 0:
            return -120.0
        return 20.0 * math.log10(x / 32768.0)

    return to_dbfs(rms), to_dbfs(peak)


def vu_bar(db, floor=-60.0, ceiling=0.0, width=40):
    """ASCII bar from floor..ceiling with a marker at current level."""
    db = max(floor, min(db, ceiling))
    frac = (db - floor) / (ceiling - floor)
    pos = int(round(frac * width))
    left = "-" * max(0, pos - 1)
    right = "-" * max(0, width - pos)
    return f"[{left}|{right}] {db:5.1f} dBFS"


class Ema:
    """Simple exponential moving average for smoothing."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        self.val = x if self.val is None else (self.alpha * x + (1 - self.alpha) * self.val)
        return self.val

WORDS_PER_SEC_EMA = Ema(alpha=0.3)  # (kept if you extend later)
RMS_DBFS_EMA      = Ema(alpha=0.3)


# ──────────────────────────────────────────────────────────────
# 2) AUDIO MONITOR (FFPLAY) — SINGLE DEFINITIONS
# ──────────────────────────────────────────────────────────────
monitor_proc = None

def start_audio_monitor(stream_url: str, monitor_cfg: dict):
    """
    Spawn ffplay to monitor the live audio.
    Supports dshow mic ("audio=...") and network streams.
    """
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        return  # already running

    vol = float(monitor_cfg.get("volume", 1.0))
    vol_filter = f"volume={vol:.2f}"

    if stream_url.startswith("audio="):
        cmd = [
            "ffplay", "-loglevel", "warning", "-nodisp",
            "-f", "dshow", "-i", stream_url,
            "-af", vol_filter,
        ]
    else:
        cmd = [
            "ffplay", "-loglevel", "warning", "-nodisp",
            "-autoexit", stream_url,
            "-af", vol_filter,
        ]

    try:
        monitor_proc = subprocess.Popen(cmd)
        print(f"[worker] 🔊 Audio monitor started (pid {monitor_proc.pid})")
    except FileNotFoundError:
        print("[worker] ⚠ ffplay not found. Install FFmpeg and ensure ffplay is on PATH.")
    except Exception as e:
        print(f"[worker] ⚠ Could not start audio monitor: {e}")


def stop_audio_monitor():
    """Cleanly stop the ffplay monitor."""
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        try:
            monitor_proc.terminate()
            monitor_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            monitor_proc.kill()
        except Exception:
            pass
        print("[worker] 🔇 Audio monitor stopped.")
    monitor_proc = None


# ──────────────────────────────────────────────────────────────
# 3) CARTRIDGE DISCOVERY & LOADING
# ──────────────────────────────────────────────────────────────
def list_cartridges():
    base = os.path.join(os.path.dirname(__file__), "cartridges")
    carts = []
    for f in os.listdir(base):
        if f.endswith(".py") and f not in ["__init__.py", "template.py"]:
            carts.append(f[:-3])
    return carts

cartridge_names = list_cartridges()
if not cartridge_names:
    print("❌ No cartridges found in ./cartridges/")
    sys.exit(1)

print(f"\n🎮 Available Cartridges:")
for i, name in enumerate(cartridge_names, 1):
    print(f" {i}. {name}")
choice = input("\nSelect cartridge number to load: ").strip()

try:
    choice_idx = int(choice) - 1
    CARTRIDGE = cartridge_names[choice_idx]
except (ValueError, IndexError):
    print("❌ Invalid selection.")
    sys.exit(1)

# Load chosen cartridge
try:
    cart = importlib.import_module(f"cartridges.{CARTRIDGE}")
except ModuleNotFoundError:
    print(f"❌ Cartridge '{CARTRIDGE}' not found.")
    sys.exit(1)

STATION_ID   = getattr(cart, "STATION_ID", "unknown_station")
DISPLAY_NAME = getattr(cart, "DISPLAY_NAME", STATION_ID)
STREAM_URL   = getattr(cart, "STREAM_URL", None)
SETTINGS     = getattr(cart, "SETTINGS", {})
AWARE_RULES  = getattr(cart, "AWARE_RULES", {})
GEO_HINTS    = getattr(cart, "GEO_HINTS", {})
VU           = getattr(cart, "VU", {"enabled": False})
MONITOR      = getattr(cart, "MONITOR", {"enabled": False, "volume": 1.0})
ENRICH       = getattr(cart, "ENRICH", {"wiki": False, "map": False, "min_chars": 3})
SYSTEM_DEVICE = getattr(cart, "SYSTEM_DEVICE", None)

if not STREAM_URL:
    print(f"❌ Cartridge '{CARTRIDGE}' does not define STREAM_URL.")
    sys.exit(1)

preprocess_text = getattr(cart, "preprocess_text", lambda t: t)
on_event_hook   = getattr(cart, "on_event", lambda e: e)

CHUNK_SECONDS   = SETTINGS.get("chunk_seconds", 4)
WHISPER_MODEL   = SETTINGS.get("whisper_model", "base.en")
WHISPER_DEVICE  = SETTINGS.get("whisper_device", "cpu")
WHISPER_COMPUTE = SETTINGS.get("whisper_compute", "int8")

DB_URL  = os.getenv("DATABASE_URL", "postgresql://postgres:radio@localhost:5432/postgres")
RDS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

print(f"\n[worker] 🎮 Loaded cartridge: {Fore.CYAN}{CARTRIDGE}{Style.RESET_ALL}")
print(f"[worker] Station: {Fore.YELLOW}{DISPLAY_NAME}{Style.RESET_ALL}")
print(f"[worker] Stream URL: {STREAM_URL}")
print(f"[worker] Settings: {SETTINGS}\n")

if MONITOR.get("enabled", False):
    start_audio_monitor(STREAM_URL, MONITOR)


# ──────────────────────────────────────────────────────────────
# 4) CONNECTIONS & MODELS
# ──────────────────────────────────────────────────────────────
print("[worker] Connecting to Postgres & Memurai…")
pg  = psycopg.connect(DB_URL)          # (You’re not using pg in this script right now; kept for parity)
rds = redis.from_url(RDS_URL)
print(f"[worker] {Fore.GREEN}Connections ready.{Style.RESET_ALL}")

print("[worker] Loading spaCy & Whisper…")
nlp   = spacy.load("en_core_web_sm")
model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print(f"[worker] {Fore.GREEN}Ready. Press Ctrl+C to stop.{Style.RESET_ALL}")


# ──────────────────────────────────────────────────────────────
# 5) AWARENESS TRACKER (per-cartridge rules)
# ──────────────────────────────────────────────────────────────
class AwarenessTracker:
    def __init__(self, rules):
        self.rules = rules
        self.counts_kw = defaultdict(int)
        self.counts_ent = defaultdict(int)

    def update(self, text, ents):
        metrics = {"keywords": {}, "entities": {}}

        # Keyword counts (configurable case-sensitivity)
        for kw, opt in self.rules.get("keywords", {}).items():
            cs = opt.get("case_sensitive", False)
            hay = text if cs else text.lower()
            needle = kw if cs else kw.lower()
            hits = len(re.findall(rf"\b{re.escape(needle)}\b", hay))
            if hits:
                self.counts_kw[needle] += hits
                metrics["keywords"][needle] = self.counts_kw[needle]

        # Entity counts by label type
        for ent in ents:
            lbl = ent.get("type") or ent.get("label")
            if self.rules.get("entities", {}).get(lbl, False):
                name = ent["text"]
                self.counts_ent[name] += 1
                metrics["entities"][name] = self.counts_ent[name]
        return metrics

AWARE = AwarenessTracker(AWARE_RULES)


# ──────────────────────────────────────────────────────────────
# 6) ENRICHMENT HELPERS (Wiki/Map) + PUBLISH
# ──────────────────────────────────────────────────────────────
ENTITY_OK_TYPES    = {"PERSON", "ORG", "GPE", "LOC"}  # editable per needs
_WIKI_CACHE        = {}   # title.lower() -> {"ts": epoch, "data": {...}}
_WIKI_TTL          = 60 * 30  # 30 min cache
_LAST_WIKI_CALL    = 0.0
_WIKI_MIN_INTERVAL = 1.2   # polite rate-limit

def _norm_title(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def _should_enrich_entity(ent_text: str, ent_label: str) -> bool:
    if not ent_text or len(ent_text) < ENRICH.get("min_chars", 3):
        return False
    if ent_label and ent_label.upper() not in ENTITY_OK_TYPES:
        return False
    if re.fullmatch(r"[0-9\-:/.]+", ent_text.strip()):  # ignore purely numeric
        return False
    return True

def _wiki_cached_get(title: str):
    v = _WIKI_CACHE.get(title.lower())
    if v and (time.time() - v["ts"] < _WIKI_TTL):
        return v["data"]
    return None

def _wiki_cached_set(title: str, data: dict):
    _WIKI_CACHE[title.lower()] = {"ts": time.time(), "data": data}

def fetch_wikipedia_summary(title: str) -> dict | None:
    """Wikipedia REST summary API. Returns {title, extract, url, thumbnail} or None."""
    global _LAST_WIKI_CALL
    title = _norm_title(title)
    if not title:
        return None

    cached = _wiki_cached_get(title)
    if cached is not None:
        return cached

    # rate-limit
    dt = time.time() - _LAST_WIKI_CALL
    if dt < _WIKI_MIN_INTERVAL:
        time.sleep(_WIKI_MIN_INTERVAL - dt)

    page = "_".join(w.capitalize() for w in title.split())
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}"
    headers = {"User-Agent": "Raidio/0.1 (self-hosted)"}

    try:
        resp = requests.get(url, headers=headers, timeout=4)
        _LAST_WIKI_CALL = time.time()
        if resp.status_code == 404:
            _wiki_cached_set(title, None)
            return None
        resp.raise_for_status()
        data = resp.json()
        out = {
            "title": data.get("title") or title,
            "extract": data.get("extract") or "",
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{page}",
            "thumbnail": (data.get("thumbnail") or {}).get("source"),
        }
        _wiki_cached_set(title, out)
        return out
    except requests.RequestException:
        return None

def build_map_info(name: str, lat: float | None = None, lon: float | None = None) -> dict:
    """Prefer explicit lat/lon; otherwise provide a map search URL."""
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return {
            "lat": float(lat),
            "lon": float(lon),
            "map_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=12/{lat}/{lon}",
        }
    q = quote_plus(_norm_title(name))
    return {"map_url": f"https://www.google.com/maps?q={q}"}

def publish_info_for_entities(rds, entities):
    """
    Publish one 'info' message per chunk if a suitable entity is found.
    The frontend listens for {"type":"info","data":{...}}.
    """
    if not (ENRICH.get("wiki") or ENRICH.get("map")):
        return

    # entities can be dicts (from extract_entities) or spaCy spans (if used directly)
    for e in entities:
        label = (e.get("type") if isinstance(e, dict) else getattr(e, "label_", "")) or ""
        text  = (e.get("text") if isinstance(e, dict) else getattr(e, "text", "")) or ""
        label = label.upper()
        text  = _norm_title(text)
        if not _should_enrich_entity(text, label):
            continue

        info_payload = {"entity": text, "label": label}

        if ENRICH.get("wiki"):
            w = fetch_wikipedia_summary(text)
            if w:
                info_payload["wiki"] = w

        if ENRICH.get("map"):
            lat = (e.get("lat") if isinstance(e, dict) else getattr(e, "lat", None))
            lon = (e.get("lon") if isinstance(e, dict) else getattr(e, "lon", None))
            info_payload["map"] = build_map_info(text, lat, lon)

        if ("wiki" in info_payload) or ("map" in info_payload):
            rds.publish(EVENT_CHANNEL, json.dumps({
                "type": "info",
                "data": info_payload,
            }, ensure_ascii=False))
            break  # one info message per chunk to limit noise


# ---- DirectShow device probing for system loopback on Windows ----
def dshow_list_devices():
    """
    Return ffmpeg -list_devices output as text (Windows only).
    """
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True
        )
        return (p.stderr or "") + (p.stdout or "")
    except Exception:
        return ""

def pick_system_loopback_device(explicit_name: str | None = None) -> str | None:
    """
    If explicit_name is provided, return it.
    Else try to discover a likely system-output capture device.
    Returns the exact DirectShow device name to pass as -i audio="NAME",
    or None if not found.
    """
    if explicit_name:
        return explicit_name

    txt = dshow_list_devices()
    # candidates in order of reliability
    candidates = [
        r'"virtual-audio-capturer"',           # screen-capture-recorder package
        r'"Stereo Mix',                        # many Realtek devices expose this
        r'"CABLE Output (VB-Audio Virtual Cable)"',
        r'"VoiceMeeter Output',                # VoiceMeeter/ Banana
    ]

    for pat in candidates:
        for line in txt.splitlines():
            if pat in line:
                # extract the quoted name
                m = re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1)

    return None

# ──────────────────────────────────────────────────────────────
# 7) CORE AUDIO / ASR / ENTITIES
# ──────────────────────────────────────────────────────────────
def ffmpeg_grab_to(tmp_path, seconds):
    """
    Record CHUNK_SECONDS of mono 16k s16 WAV to tmp_path from:
      - Windows mic: STREAM_URL startswith("audio=") via dshow
      - Network streams: http/https/icecast
      - System output: STREAM_URL == "system" (Windows loopback)
    """
    if STREAM_URL == "system":
        # Pick a loopback device
        dev = pick_system_loopback_device(SYSTEM_DEVICE)
        if not dev:
            print("[worker] ❌ Could not find a system loopback device.")
            print("        Options:")
            print("        • Install 'VB-Audio Virtual Cable' and set SYSTEM_DEVICE='CABLE Output (VB-Audio Virtual Cable)'")
            print("        • Enable 'Stereo Mix' in Windows Sound Control Panel and set SYSTEM_DEVICE to its exact name")
            print("        • Install 'screen-capture-recorder' (virtual-audio-capturer)")
            raise RuntimeError("No system loopback device found")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "dshow",
            "-i", f"audio={dev}",
            "-t", str(seconds),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            tmp_path,
        ]

    elif STREAM_URL.startswith("audio="):  # Windows microphone via DirectShow
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "dshow", "-i", STREAM_URL,
            "-t", str(seconds),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            tmp_path,
        ]

    else:  # http(s) / icecast stream
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", STREAM_URL,
            "-t", str(seconds),
            "-vn",
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            tmp_path,
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def transcribe_file(path):
    segments, _ = model.transcribe(
        path, beam_size=4, vad_filter=True, condition_on_previous_text=False
    )
    return [s.text.strip() for s in segments if s.text.strip()]


def extract_entities(text):
    doc = nlp(text)
    return [{"text": e.text, "type": e.label_} for e in doc.ents]


def publish(rds, event):
    rds.publish(EVENT_CHANNEL, json.dumps(event, ensure_ascii=False))


# ──────────────────────────────────────────────────────────────
# 8) MAIN LOOP
# ──────────────────────────────────────────────────────────────
def main():
    last_text = ""
    vu_bar_str = ""
    vu_hint = ""
    avg_rms = None

    # Live UI
    with Live(render_ui(vu_bar_str, vu_hint, avg_rms), refresh_per_second=10, screen=False) as live:
        while True:
            t0 = datetime.datetime.utcnow()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                ffmpeg_grab_to(tmp.name, CHUNK_SECONDS)

            # Measure VU before ASR (so it's always based on actual audio)
            if VU.get("enabled", False):
                rms_dbfs, peak_dbfs = wav_rms_peak_dbfs(tmp.name)
                avg_rms = RMS_DBFS_EMA.update(rms_dbfs)
                vu_bar_str = vu_bar(
                    rms_dbfs,
                    VU.get("bar_floor", -60.0),
                    VU.get("bar_ceiling", 0.0),
                    width=40,
                )
                low    = VU.get("warn_low", -35.0)
                high   = VU.get("warn_high", -6.0)
                target = VU.get("target_dbfs", -20.0)
                if avg_rms is not None:
                    if avg_rms < low:
                        vu_hint = f"⚠ too quiet (avg {avg_rms:.1f} dBFS; target {target} dBFS)"
                    elif avg_rms > high:
                        vu_hint = f"⚠ clipping risk (avg {avg_rms:.1f} dBFS; target {target} dBFS)"
                    else:
                        vu_hint = f"✓ good level (avg {avg_rms:.1f} dBFS)"
                # redraw header immediately
                live.update(render_ui(vu_bar_str, vu_hint, avg_rms), refresh=True)

            # Transcribe
            texts = transcribe_file(tmp.name)

            for text in texts:
                text = preprocess_text(text)
                if text.strip() == last_text.strip():
                    continue
                last_text = text

                # NER & awareness
                ents    = extract_entities(text)
                metrics = AWARE.update(text, ents)

                # Update terminal transcript log (colored if entities found)
                ts     = t0.strftime("%H:%M:%S")
                prefix = "[cyan]" if ents else "[white]"
                TRANSCRIPT_LOG.append(f"{prefix}[{ts}] {text}[/]")

                # Publish the base transcript event (legacy/compatible format)
                event = {
                    "station_id": STATION_ID,
                    "chunk_id": str(uuid.uuid4()),
                    "t0": t0.isoformat() + "Z",
                    "t1": (t0 + datetime.timedelta(seconds=CHUNK_SECONDS)).isoformat() + "Z",
                    "text": text,
                    "entities": ents,
                    "awareness": metrics,
                }
                if VU.get("enabled", False):
                    event["vu"] = {
                        "rms_dbfs": float(RMS_DBFS_EMA.val) if RMS_DBFS_EMA.val is not None else None,
                        "target_dbfs": VU.get("target_dbfs", -20.0),
                        "warn_low": VU.get("warn_low", -35.0),
                        "warn_high": VU.get("warn_high", -6.0),
                    }

                # Hook for per-cartridge tweaks, then publish the transcript
                event = on_event_hook(event)
                publish(rds, event)

                # ✅ NEW: publish enrichment "info" event (Wiki/Map) if enabled
                publish_info_for_entities(rds, ents)

                # Repaint UI
                live.update(render_ui(vu_bar_str, vu_hint, avg_rms), refresh=True)

            # Pace loop slightly under CHUNK_SECONDS
            time.sleep(max(0, CHUNK_SECONDS - 0.5))


# ──────────────────────────────────────────────────────────────
# 9) ENTRYPOINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] Stopping…")
    finally:
        stop_audio_monitor()
