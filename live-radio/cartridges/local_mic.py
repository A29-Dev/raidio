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
