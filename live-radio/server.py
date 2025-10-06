# server.py
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis

# ── Config (env overrides) ─────────────────────────────────────────────────────
REDIS_URL     = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "radio_events")
WS_PATH       = os.getenv("WS_PATH", "/live")  # path the browser connects to

# ── App & CORS (allow local dev UIs) ──────────────────────────────────────────
app = FastAPI(title="Raidio Realtime Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# One shared Redis client (text mode)
r = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "ok": True,
        "info": "Raidio FastAPI bridge is running.",
        "ws": WS_PATH,
        "redis": REDIS_URL,
        "channel": EVENT_CHANNEL,
    }

@app.get("/healthz")
async def healthz():
    # Basic health + one Redis round-trip
    try:
        pong = await r.ping()
    except Exception as e:
        pong = f"error: {e!r}"
    return {"ok": True, "redis": REDIS_URL, "channel": EVENT_CHANNEL, "ping": pong}

# WebSocket endpoint (mounted at WS_PATH, default /live)
@app.websocket(WS_PATH)
async def live(ws: WebSocket):
    await ws.accept()
    print(f"[server] WS connect from client")
    pubsub = r.pubsub()
    await pubsub.subscribe(EVENT_CHANNEL)
    print(f"[server] subscribed to '{EVENT_CHANNEL}' on {REDIS_URL}")

    # Heartbeat so intermediaries don't drop the socket
    async def heartbeat():
        while True:
            await asyncio.sleep(25)
            try:
                await ws.send_text('{"type":"ping"}')
            except Exception:
                break

    hb_task = asyncio.create_task(heartbeat())

    try:
        async for msg in pubsub.listen():
            if not msg or msg.get("type") != "message":
                continue
            data = msg.get("data")
            if data is None:
                continue
            # Ensure JSON/text
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", "ignore")

            # Optional tiny log to prove forwarding
            if isinstance(data, str):
                preview = data[:80].replace("\n", " ")
                print(f"[server] → WS {len(data)} bytes | {preview!r}")

            await ws.send_text(data)

    except WebSocketDisconnect:
        print("[server] WS disconnected")
    except Exception as e:
        print(f"[server] WS error: {e!r}")
    finally:
        hb_task.cancel()
        try:
            await pubsub.unsubscribe(EVENT_CHANNEL)
            await pubsub.close()
            print(f"[server] unsubscribed from '{EVENT_CHANNEL}'")
        except Exception as e:
            print("[server] pubsub close err:", e)
        try:
            await ws.close()
        except Exception:
            pass


# ── Dev entrypoint (uvicorn) ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
