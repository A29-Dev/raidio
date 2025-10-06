from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncpg, redis.asyncio as aioredis, asyncio, json
import os
import redis.asyncio as aioredis



DB_URL = "postgresql://radio:radio@localhost:5432/radio"
REDIS_URL = "redis://127.0.0.1:6379/0"     # <── Memurai address
r = aioredis.from_url("redis://localhost:6379", decode_responses=True)
EVENT_CHANNEL = "radio_events"

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/stations", "WS /live", "/docs"]}


@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(DB_URL)
    app.state.rds = aioredis.from_url(REDIS_URL, decode_responses=True)

@app.get("/stations")
async def stations():
    async with app.state.pool.acquire() as con:
        rows = await con.fetch("SELECT id,name,stream_url FROM stations;")
    return [dict(r) for r in rows]

@app.websocket("/live")
async def live(ws: WebSocket):
    print(f"[server] WS connect from client")
    await ws.accept()
    pubsub = r.pubsub()
    await pubsub.subscribe(EVENT_CHANNEL)
    print(f"[server] subscribed to '{EVENT_CHANNEL}' on {REDIS_URL}")
    try:
        async for msg in pubsub.listen():
            if msg and msg.get("type") == "message":
                data = msg["data"]
                # ensure text
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", "ignore")
                await ws.send_text(data)
    except WebSocketDisconnect:
        print("[server] WS disconnected")
    finally:
        try:
            await pubsub.unsubscribe(EVENT_CHANNEL)
            await pubsub.close()
            print(f"[server] unsubscribed from '{EVENT_CHANNEL}'")
        except Exception as e:
            print("[server] pubsub close err:", e)
        try:
            await ws.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", port=8000, reload=True)
