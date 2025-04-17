import asyncio
import os
import random
import socket
import threading
import uuid

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request


# ─── Load Manager ────────────────────────────────────────────────────────────
class LoadManager:
    """Track number of in-flight tasks per instance and pick the least-loaded one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._load: dict[str, int] = {}

    def register(self, addr: str):
        """Ensure this addr is known, with zero initial load."""
        with self._lock:
            self._load.setdefault(addr, 0)

    def acquire(self) -> str:
        """Pick the addr with minimal load and bump its count."""
        with self._lock:
            if not self._load:
                raise RuntimeError("No instances registered")
            # find instance with smallest load
            addr = min(self._load.items(), key=lambda kv: kv[1])[0]
            self._load[addr] += 1
            return addr

    def release(self, addr: str):
        """Decrement the load count for this addr."""
        with self._lock:
            if addr in self._load and self._load[addr] > 0:
                self._load[addr] -= 1


# managers for prefill and decode pools
prefill_load_manager = LoadManager()
decode_load_manager = LoadManager()

# ─── Service Discovery ───────────────────────────────────────────────────────
prefill_instances: dict[str, str] = {}  # http_address -> zmq_address
decode_instances: dict[str, str] = {}  # http_address -> zmq_address

prefill_cv = threading.Condition()
decode_cv = threading.Condition()


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            addr = data["http_address"]
            zmq_addr = data["zmq_address"]
            if data["type"] == "P":
                with prefill_cv:
                    prefill_instances[addr] = zmq_addr
                    prefill_load_manager.register(addr)
            elif data["type"] == "D":
                with decode_cv:
                    decode_instances[addr] = zmq_addr
                    decode_load_manager.register(addr)
            else:
                print(f"Unexpected message type from {remote_address}: {data}")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    listener = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    listener.start()
    return listener


# ─── HTTP Proxy ───────────────────────────────────────────────────────────────
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


def random_uuid() -> str:
    return uuid.uuid4().hex


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(1024):
                yield chunk


async def _stream_and_release(gen, manager: LoadManager, addr: str):
    try:
        async for chunk in gen:
            yield chunk
    finally:
        manager.release(addr)


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    original = await request.get_json()
    prefill_data = original.copy()
    prefill_data["max_tokens"] = 1

    # pick least-loaded prefill
    with prefill_cv:
        prefill_addr = prefill_load_manager.acquire()
        prefill_zmq = prefill_instances[prefill_addr]
        print(f"Selected prefill {prefill_addr} (load bumped)")

    # finish prefill stage
    prefill_req_id = f"pre_{random_uuid()}"
    async for _ in forward_request(
        f"http://{prefill_addr}/v1/completions", prefill_data, prefill_req_id
    ):
        pass
    # release prefill slot
    prefill_load_manager.release(prefill_addr)

    # pick least-loaded decode
    with decode_cv:
        decode_addr = decode_load_manager.acquire()
        decode_zmq = decode_instances[decode_addr]
        print(f"Selected decode {decode_addr} (load bumped)")

    # stream decode back to client, releasing when done
    decode_req_id = f"dec_{random_uuid()}"
    decoder = forward_request(
        f"http://{decode_addr}/v1/completions", original, decode_req_id
    )
    wrapped = _stream_and_release(decoder, decode_load_manager, decode_addr)
    response = await make_response(wrapped)
    response.timeout = None
    return response


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_chat_request():
    original = await request.get_json()
    prefill_data = original.copy()
    prefill_data["max_tokens"] = 1

    with prefill_cv:
        prefill_addr = prefill_load_manager.acquire()
        prefill_zmq = prefill_instances[prefill_addr]
        print(f"Selected prefill(chat) {prefill_addr}")

    prefill_req_id = f"pre_chat_{random_uuid()}"
    async for _ in forward_request(
        f"http://{prefill_addr}/v1/chat/completions",
        prefill_data,
        prefill_req_id,
    ):
        pass
    prefill_load_manager.release(prefill_addr)

    with decode_cv:
        decode_addr = decode_load_manager.acquire()
        decode_zmq = decode_instances[decode_addr]
        print(f"Selected decode(chat) {decode_addr}")

    decode_req_id = f"dec_chat_{random_uuid()}"
    decoder = forward_request(
        f"http://{decode_addr}/v1/chat/completions", original, decode_req_id
    )
    wrapped = _stream_and_release(decoder, decode_load_manager, decode_addr)
    response = await make_response(wrapped)
    response.timeout = None
    return response


if __name__ == "__main__":
    start_service_discovery("0.0.0.0", 30001)
    app.run(host="0.0.0.0", port=10001)
