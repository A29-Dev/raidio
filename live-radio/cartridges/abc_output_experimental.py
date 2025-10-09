STATION_ID = "abc_perth"
DISPLAY_NAME = "ABC Radio National WA"
STREAM_URL = "https://live-radio02.mediahubaustralia.com/2RNW/mp3"

SETTINGS = {
    "chunk_seconds": 10,
    "whisper_model": "base.en",
    "whisper_device": "cuda",
    "whisper_compute": "float32",
}

MONITOR = {
    "enabled": True,   # True = play through speakers, False = silent
    "volume": 1.0,     # 1.0 = normal volume, 0.5 = quieter, 2.0 = +6dB boost
}


VU = {
    "enabled": True,       # turn meter on/off for this station
    "target_dbfs": -20.0,  # “sweet spot” average loudness (RMS)
    "warn_low": -35.0,     # below this = likely too quiet for good ASR
    "warn_high": -6.0,     # above this = risk of clipping/distortion
    "bar_floor": -60.0,    # left edge of the VU bar
    "bar_ceiling": 0.0,    # right edge (0 dBFS is full scale)
}

ENRICH = {
    "wiki": True,    # fetch Wikipedia summary for PERSON/ORG/GPE/LOC
    "map":  True,    # include a map URL for the entity
    "min_chars": 3,  # ignore super-short tokens
}

AWARE_RULES = {
    "keywords": {
        "budget": {"case_sensitive": False},
        "election": {"case_sensitive": False},
        "parliament": {"case_sensitive": False},
        "Emergency": {"case_sensitive": False},
    },
    "entities": {"PERSON": True},
}

def preprocess_text(text: str) -> str:
    return text

def on_event(event: dict) -> dict:
    event.setdefault("tags", []).append("abc_rn")
    return event
