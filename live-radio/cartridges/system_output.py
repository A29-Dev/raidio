# cartridges/system_output.py
# Listen to Windows system output (loopback)

STATION_ID   = "system_out"
DISPLAY_NAME = "System Output (Loopback)"
STREAM_URL = "audio=Stereo Mix (Realtek(R) Audio)"  # ← special value handled by worker

# OPTIONAL: if auto-detect fails, put your exact DirectShow device name here
# Examples:
#   "virtual-audio-capturer"                  (screen-capture-recorder pack)
#   "Stereo Mix (Realtek(R) Audio)"
#   "CABLE Output (VB-Audio Virtual Cable)"
SYSTEM_DEVICE = None

SETTINGS = {
    "chunk_seconds": 6,
    "whisper_model": "base.en",
    "whisper_device": "cuda",
    "whisper_compute": "float16",
}

# Don't start a monitor for loopback (avoids audio feedback)
MONITOR = {"enabled": False}

# Keep your enrichment toggles if you want wiki/map cards
ENRICH = {"wiki": True, "map": True, "min_chars": 3}

# Optional awareness rules
AWARE_RULES = {
    "keywords": {"youtube": {"case_sensitive": False}},
    "entities": {"PERSON": True, "GPE": True, "ORG": True},
}
