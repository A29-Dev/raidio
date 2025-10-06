# ──────────────────────────────────────────────────────────────
# 🎙️  Local Microphone Cartridge (for testing)
# ──────────────────────────────────────────────────────────────

STATION_ID = "local_mic"
DISPLAY_NAME = "Local Microphone Input"
STREAM_URL = "audio=Headset Microphone (2- SteelSeries Arctis Nova 5X)"  # special flag handled by worker.py

SETTINGS = {
    "chunk_seconds": 10,          # longer chunk = clearer speech capture
    "whisper_model": "base.en",  # small or base.en for real-time
    "whisper_device": "cuda",     # or "cuda" if you fixed GPU libs
    "whisper_compute": "float16",
}

VU = {
    "enabled": True,       # turn meter on/off for this station
    "target_dbfs": -20.0,  # “sweet spot” average loudness (RMS)
    "warn_low": -35.0,     # below this = likely too quiet for good ASR
    "warn_high": -6.0,     # above this = risk of clipping/distortion
    "bar_floor": -60.0,    # left edge of the VU bar
    "bar_ceiling": 0.0,    # right edge (0 dBFS is full scale)
}

# Awareness / keyword tracking rules
AWARE_RULES = {
    "keywords": {
        "test": {"case_sensitive": False},
        "hello": {"case_sensitive": False},
        "radio": {"case_sensitive": False},
        "keyword": {"case_sensitive": False},
    },
    "entities": {"PERSON": True},
}

def preprocess_text(text: str) -> str:
    for kw in ("test", "hello", "radio", "keyword"):
        if kw in text.lower():
            print(f"⚡ Detected keyword: {kw}")
    return text


def on_event(event: dict) -> dict:
    """Modify or tag the final event before it's published."""
    event.setdefault("tags", []).append("mic_input")
    return event
