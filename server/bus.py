"""WebSocket broadcaster + event persistence.

Every agent/watcher event goes through emit_event(): persisted to Mongo and
pushed to all connected dashboards in one call.
"""

import asyncio
from datetime import datetime, timezone

from fastapi import WebSocket

from . import db


class Broadcaster:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)


bus = Broadcaster()


def now() -> datetime:
    return datetime.now(timezone.utc)


async def emit_event(
    type_: str, message: str, run_id: str | None = None, payload: dict | None = None
) -> dict:
    doc = {
        "ts": now(),
        "run_id": run_id,
        "type": type_,
        "message": message,
        "payload": payload or {},
    }
    await db.events.insert_one({**doc})
    wire = {**doc, "ts": doc["ts"].isoformat()}
    wire.pop("_id", None)
    await bus.broadcast({"type": "event", "event": wire})
    return doc
