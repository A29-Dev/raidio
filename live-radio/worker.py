import os, sys, time, json, uuid, importlib, tempfile, subprocess, datetime
import redis, psycopg
import wave, array, math
from faster_whisper import WhisperModel
import spacy
import subprocess, os, sys, signal
from collections import deque
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from colorama import Fore, Style, init as colorama_init


EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "radio_events")

#region VU METER / UI

colorama_init()

LOG_MAX = 80
TRANSCRIPT_LOG = deque(maxlen=LOG_MAX)

def render_ui(vu_bar_str: str, vu_hint: str, avg_rms: float | None):
    # Header (VU) at top
    header_text = Text(vu_bar_str or "", style="bold green")
    if vu_hint:
        header_text.append(f"\n{vu_hint}", style="yellow")
    if avg_rms is not None:
        header_text.append(f"\nAvg level: {avg_rms:0.1f} dBFS", style="cyan")

    header = Panel(header_text, title="🎚️ Live VU", border_style="green", padding=(1,2))

    # Transcript table below
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
    # RMS
    acc = 0
    for s in samples:
        acc += s * s
    rms = math.sqrt(acc / len(samples))
    # Convert to dBFS (0 dBFS == max 32768)
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
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        self.val = x if self.val is None else (self.alpha * x + (1 - self.alpha) * self.val)
        return self.val

WORDS_PER_SEC_EMA = Ema(alpha=0.3)
RMS_DBFS_EMA = Ema(alpha=0.3)
#endregion

#region audio monitor process
monitor_proc = None

def start_audio_monitor(stream_url: str, monitor_cfg: dict):
    """Spawn ffplay to monitor the live audio."""
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        return  # already running

    vol = float(monitor_cfg.get("volume", 1.0))
    vol_filter = f"volume={vol:.2f}"

    if stream_url.startswith("audio="):
        # Windows mic device
        cmd = [
            "ffplay",
            "-loglevel", "warning",
            "-nodisp",
            "-f", "dshow",
            "-i", stream_url,
            "-af", vol_filter,
        ]
    else:
        # Network stream
        cmd = [
            "ffplay",
            "-loglevel", "warning",
            "-nodisp",
            "-autoexit",
            stream_url,
            "-af", vol_filter,
        ]

    try:
        monitor_proc = subprocess.Popen(cmd)
        print(f"[worker] 🔊 Audio monitor started (pid {monitor_proc.pid})")
    except FileNotFoundError:
        print("[worker] ⚠ ffplay not found. Make sure FFmpeg is installed and on PATH.")
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
        print("[worker] 🔇 Audio monitor stopped.")
    monitor_proc = None
#endregion

# ──────────────────────────────────────────────────────────────
# 1. FIND AVAILABLE CARTRIDGES
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



    

# ──────────────────────────────────────────────────────────────
# 2. LOAD THE CHOSEN CARTRIDGE
# ──────────────────────────────────────────────────────────────
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

VU = getattr(cart, "VU", {"enabled": False})

if not STREAM_URL:
    print(f"❌ Cartridge '{CARTRIDGE}' does not define STREAM_URL.")
    sys.exit(1)

preprocess_text = getattr(cart, "preprocess_text", lambda t: t)
on_event_hook   = getattr(cart, "on_event", lambda e: e)

CHUNK_SECONDS   = SETTINGS.get("chunk_seconds", 4)
WHISPER_MODEL   = SETTINGS.get("whisper_model", "base.en")
WHISPER_DEVICE  = SETTINGS.get("whisper_device", "cpu")
WHISPER_COMPUTE = SETTINGS.get("whisper_compute", "int8")

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:radio@localhost:5432/postgres")
RDS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

MONITOR = getattr(cart, "MONITOR", {"enabled": False, "volume": 1.0})

print(f"\n[worker] 🎮 Loaded cartridge: {Fore.CYAN}{CARTRIDGE}{Style.RESET_ALL}")
print(f"[worker] Station: {Fore.YELLOW}{DISPLAY_NAME}{Style.RESET_ALL}")
print(f"[worker] Stream URL: {STREAM_URL}")
print(f"[worker] Settings: {SETTINGS}\n")



if MONITOR.get("enabled", False):
    start_audio_monitor(STREAM_URL, MONITOR)





# ──────────────────────────────────────────────────────────────
# 3. SETUP CONNECTIONS
# ──────────────────────────────────────────────────────────────
print("[worker] Connecting to Postgres & Memurai…")
pg = psycopg.connect(DB_URL)
rds = redis.from_url(RDS_URL)
print(f"[worker] {Fore.GREEN}Connections ready.{Style.RESET_ALL}")

print("[worker] Loading spaCy & Whisper…")
nlp = spacy.load("en_core_web_sm")
model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print(f"[worker] {Fore.GREEN}Ready. Press Ctrl+C to stop.{Style.RESET_ALL}")

# ──────────────────────────────────────────────────────────────
# 4. AWARENESS TRACKER
# ──────────────────────────────────────────────────────────────
import re, collections
from collections import defaultdict

class AwarenessTracker:
    def __init__(self, rules):
        self.rules = rules
        self.counts_kw = defaultdict(int)
        self.counts_ent = defaultdict(int)

    def update(self, text, ents):
        metrics = {"keywords": {}, "entities": {}}
        for kw, opt in self.rules.get("keywords", {}).items():
            cs = opt.get("case_sensitive", False)
            hay = text if cs else text.lower()
            needle = kw if cs else kw.lower()
            hits = len(re.findall(rf"\b{re.escape(needle)}\b", hay))
            if hits:
                self.counts_kw[needle] += hits
                metrics["keywords"][needle] = self.counts_kw[needle]

        for ent in ents:
            lbl = ent.get("type") or ent.get("label")
            if self.rules.get("entities", {}).get(lbl, False):
                name = ent["text"]
                self.counts_ent[name] += 1
                metrics["entities"][name] = self.counts_ent[name]
        return metrics

AWARE = AwarenessTracker(AWARE_RULES)

# ──────────────────────────────────────────────────────────────
# 5. CORE FUNCTIONS
# ──────────────────────────────────────────────────────────────
def ffmpeg_grab_to(tmp_path, seconds):
    # Detect Windows microphone device
    if STREAM_URL.startswith("audio="):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "dshow",               # ← this tells FFmpeg it's a Windows audio device
            "-i", STREAM_URL,
            "-t", str(seconds),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            tmp_path,
        ]
    else:
        # Normal stream input
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", STREAM_URL,
            "-t", str(seconds),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            tmp_path,
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_file(path):
    segments, _ = model.transcribe(path, beam_size=1, vad_filter=True, condition_on_previous_text=False)
    return [s.text.strip() for s in segments if s.text.strip()]

def extract_entities(text):
    doc = nlp(text)
    return [{"text": e.text, "type": e.label_} for e in doc.ents]

def publish(rds, event):
    rds.publish(EVENT_CHANNEL, json.dumps(event, ensure_ascii=False))

# ──────────────────────────────────────────────────────────────
# 6. MAIN LOOP
# ──────────────────────────────────────────────────────────────


monitor_proc = None

def start_audio_monitor(stream_url: str, monitor_cfg: dict):
    """
    Spawns an ffplay process to monitor the audio in real time.
    - Supports http(s) streams and Windows dshow microphones ("audio=...").
    """
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        return  # already running

    vol = float(monitor_cfg.get("volume", 1.0))
    vol_filter = f"volume={vol:.2f}"

    if stream_url.startswith("audio="):
        # Windows microphone via DirectShow
        cmd = [
            "ffplay",
            "-loglevel", "warning",
            "-nodisp",
            "-f", "dshow",
            "-i", stream_url,
            "-af", vol_filter,
        ]
    else:
        # Network stream (http/s, icecast)
        cmd = [
            "ffplay",
            "-loglevel", "warning",
            "-nodisp",
            "-autoexit",  # ends if source ends; for live it keeps running
            stream_url,
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
    global monitor_proc
    if monitor_proc and monitor_proc.poll() is None:
        try:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()
        except Exception:
            pass
        finally:
            print("[worker] 🔇 Audio monitor stopped.")
    monitor_proc = None

def main():
    last_text = ""
    vu_bar_str = ""
    vu_hint = ""
    avg_rms = None

    # Start a live UI that we can update continuously
    with Live(render_ui(vu_bar_str, vu_hint, avg_rms), refresh_per_second=10, screen=False) as live:
        while True:
            t0 = datetime.datetime.utcnow()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                ffmpeg_grab_to(tmp.name, CHUNK_SECONDS)

            # --- VU (measure BEFORE transcribe)
            if VU.get("enabled", False):
                rms_dbfs, peak_dbfs = wav_rms_peak_dbfs(tmp.name)
                avg_rms = RMS_DBFS_EMA.update(rms_dbfs)
                vu_bar_str = vu_bar(
                    rms_dbfs,
                    VU.get("bar_floor", -60.0),
                    VU.get("bar_ceiling", 0.0),
                    width=40,
                )
                low   = VU.get("warn_low", -35.0)
                high  = VU.get("warn_high", -6.0)
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

            texts = transcribe_file(tmp.name)

            for text in texts:
                text = preprocess_text(text)
                if text.strip() == last_text.strip():
                    continue
                last_text = text

                ents = extract_entities(text)
                metrics = AWARE.update(text, ents)

                # add a colored line to the transcript log
                ts = t0.strftime("%H:%M:%S")
                prefix = "[cyan]" if ents else "[white]"
                TRANSCRIPT_LOG.append(f"{prefix}[{ts}] {text}[/]")

                # (publish event as you already do)
                event = {
                    "station_id": STATION_ID,
                    "chunk_id": str(uuid.uuid4()),
                    "t0": t0.isoformat()+"Z",
                    "t1": (t0 + datetime.timedelta(seconds=CHUNK_SECONDS)).isoformat()+"Z",
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
                event = on_event_hook(event)
                publish(rds, event)

                # redraw both header + updated transcript
                live.update(render_ui(vu_bar_str, vu_hint, avg_rms), refresh=True)

            time.sleep(max(0, CHUNK_SECONDS - 0.5))

# ──────────────────────────────────────────────────────────────
# 7. ENTRYPOINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] Stopping…")
    finally:
        stop_audio_monitor()

