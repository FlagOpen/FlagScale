from __future__ import annotations

import asyncio
import os
import random
import socket
import threading
import uuid
from enum import Enum
from typing import Dict, Literal, Optional, Tuple

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request


# ─── Registry ──────────────────────────────────────────────────────────────────
class InstanceType(Enum):
    PREFILL = "P"
    DECODE = "D"


class ServiceRegistry:
    """Thread‑safe container for prefill / decode instance metadata."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cond_prefill = threading.Condition(self._lock)
        self._cond_decode = threading.Condition(self._lock)
        self._instances: Dict[InstanceType, Dict[str, str]] = {
            InstanceType.PREFILL: {},
            InstanceType.DECODE: {},
        }

    # ---- public API ----------------------------------------------------------
    def register(
        self,
        itype: InstanceType | Literal["P", "D"],
        http_addr: str,
        zmq_addr: str,
    ) -> None:
        itype = InstanceType(itype)  # cast if literal
        with self._lock:
            self._instances[itype][http_addr] = zmq_addr
            # wake one waiter if someone is blocked on this pool
            cond = (
                self._cond_prefill
                if itype is InstanceType.PREFILL
                else self._cond_decode
            )
            cond.notify()

    def random_instance(self, itype: InstanceType) -> Tuple[str, str]:
        """Blocks until at least one instance of *itype* is present."""
        cond = (
            self._cond_prefill if itype is InstanceType.PREFILL else self._cond_decode
        )
        with cond:
            while not self._instances[itype]:
                cond.wait()
            http, zmq_ = random.choice(list(self._instances[itype].items()))
            return http, zmq_

    def size(self, itype: InstanceType) -> int:
        with self._lock:
            return len(self._instances[itype])


# ─── ZMQ listener -------------------------------------------------------------
def _listen_for_register(registry: ServiceRegistry, router_socket, poller) -> None:
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            _, message = router_socket.recv_multipart()
            data = msgpack.loads(
                message
            )  # {"type":"P","http_address":..,"zmq_address":..}
            try:
                registry.register(
                    data["type"], data["http_address"], data["zmq_address"]
                )
            except (KeyError, ValueError):
                print("⚠️  malformed registration data:", data)


def start_service_discovery(
    registry: ServiceRegistry, hostname: str, port: int
) -> threading.Thread:
    if port == 0:
        raise ValueError("Port cannot be 0")
    if not hostname:
        hostname = socket.gethostname()

    ctx, router = zmq.Context(), zmq.Context().socket(zmq.ROUTER)
    router.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    t = threading.Thread(
        target=_listen_for_register, daemon=True, args=(registry, router, poller)
    )
    t.start()
    return t


# ─── HTTP proxy server --------------------------------------------------------
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)
registry = ServiceRegistry()  # <── NEW single global


def _uuid() -> str:
    return uuid.uuid4().hex


async def forward_request(url: str, data: dict, request_id: str):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as sess:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
        }
        async with sess.post(url, json=data, headers=headers) as resp:
            if resp.status == 200:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk
            else:
                raise RuntimeError(f"Upstream {url} returned {resp.status}")


async def _handle_common(original_request: dict, api_path: str):
    # pick instances
    pre_http, pre_zmq = registry.random_instance(InstanceType.PREFILL)
    dec_http, dec_zmq = registry.random_instance(InstanceType.DECODE)
    request_id = f"___prefill_{pre_zmq}___decode_{dec_zmq}___{_uuid()}"

    # 1️⃣   prefill: max_tokens = 1
    prefill_request = {**original_request, "max_tokens": 1}
    async for _ in forward_request(
        f"http://{pre_http}{api_path}", prefill_request, request_id
    ):
        continue

    # 2️⃣   decode: stream back to client
    generator = forward_request(
        f"http://{dec_http}{api_path}", original_request, request_id
    )
    resp = await make_response(generator)
    resp.timeout = None
    return resp


@app.post("/v1/completions")
async def handle_request():  # legacy openai completions
    return await _handle_common(await request.get_json(), "/v1/completions")


@app.post("/v1/chat/completions")
async def handle_chat_request():  # chat completions
    return await _handle_common(await request.get_json(), "/v1/chat/completions")


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    discovery_thread = start_service_discovery(registry, "0.0.0.0", 30001)
    try:
        # Quart uses asyncio, so run() is non‑blocking once the event loop starts.
        app.run(host="0.0.0.0", port=10001)
    finally:
        discovery_thread.join()
