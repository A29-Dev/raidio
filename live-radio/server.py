from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncpg, redis.asyncio as aioredis, asyncio, json

app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_URL = "postgresql://radio:radio@localhost:5432/radio"
REDIS_URL = "redis://127.0.0.1:6379/0"     # <── Memurai address

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
    await ws.accept()
    sub = app.state.rds.pubsub()
    await sub.subscribe("events")
    try:
        async for msg in sub.listen():
            if msg.get("type") == "message":
                await ws.send_text(msg["data"])
    finally:
        await sub.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", port=8000, reload=True)
