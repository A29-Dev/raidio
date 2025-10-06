# --- Station "cartridge" template -------------------------------------------

# Metadata
STATION_ID = "example_station"
DISPLAY_NAME = "Example Station"

# Worker knobs (override defaults)
SETTINGS = {
    "chunk_seconds": 4,           # ffmpeg grab length
    "whisper_model": "base.en",   # tiny/base.en/small.en/…
    "whisper_device": "cpu",      # cpu/cuda
    "whisper_compute": "int8",    # int8/float16/…
}

# AU/Local geo bias (add as you like)
GEO_HINTS = {
    "Fremantle": (-32.0569, 115.7439),
    "Cottesloe": (-31.9940, 115.7510),
}

# Awareness rules (ephemeral, in-memory)
AWARE_RULES = {
    "keywords": {                # whole-word counts
        "the": {"case_sensitive": False},
        "budget": {"case_sensitive": False},
    },
    "entities": {                # track counts for labels
        "PERSON": True,
        "ORG": False,
    },
    "regex": [r"\bCOVID-?19\b"],
    "alerts": [
        {"type": "keyword", "term": "budget", "threshold": 3, "window_sec": 60},
        {"type": "entity",  "label": "PERSON", "threshold": 5, "window_sec": 120},
    ],
}

# Optional text hook before NLP
def preprocess_text(text: str) -> str:
    # e.g., strip DJ tag lines, normalize fillers, etc.
    return text

# Optional event hook before publish (mutate/augment)
def on_event(event: dict) -> dict:
    # e.g., add station-specific tags
    event.setdefault("tags", []).append("news")
    return event
# ---------------------------------------------------------------------------
