STATION_ID = "ABC_Goldfields_Esperance_Output"
DISPLAY_NAME = "ABC Goldfields Esperance"
STREAM_URL = "https://live-radio02.mediahubaustralia.com/6KLGW/mp3/;"

SETTINGS = {
    "chunk_seconds": 5,
    "whisper_model": "base.en",
    "whisper_device": "cuda",
    "whisper_compute": "float16",
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

AWARE_RULES = {
    "keywords": {
        "Esperance": {"case_sensitive": False},
    },
    "entities": {"PERSON": True},
}

def preprocess_text(text: str) -> str:
    return text

def on_event(event: dict) -> dict:
    event.setdefault("tags", []).append("abc_rn")
    return event
