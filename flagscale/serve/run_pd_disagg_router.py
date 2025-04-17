import os
import random
import socket
import threading
import uuid

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request


class ResourceManager:
    """Thread-safe manager for prefill and decode instances."""
    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {
            "P": {},  # type: http_address -> zmq_address
            "D": {},  # decode: http_address -> zmq_address
        }
        self._conds = {
            "P": threading.Condition(self._lock),
            "D": threading.Condition(self._lock),
        }

    def register(self, itype: str, http_addr: str, zmq_addr: str):
        """Register a new instance of type P or D."""
        with self._lock:
            self._instances[itype][http_addr] = zmq_addr
            self._conds[itype].notify_all()

    def get_random(self, itype: str) -> tuple[str, str]:
        """Get a random available instance, blocking until one is available."""
        cond = self._conds[itype]
        with cond:
            while not self._instances[itype]:
                cond.wait()
            items = list(self._instances[itype].items())
            http_addr, zmq_addr = random.choice(items)
            return http_addr, zmq_addr


def _listen_for_register(poller, router_socket, manager: ResourceManager):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            itype = data.get("type")
            if itype in ("P", "D"):
                manager.register(itype, data["http_address"], data["zmq_address"])
            else:
                print(f"Unexpected message from {remote_address}: {data}")


def start_service_discovery(hostname: str, port: int, manager: ResourceManager) -> threading.Thread:
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
        target=_listen_for_register,
        args=(poller, router_socket, manager),
        daemon=True
    )
    listener.start()
    return listener


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)
resource_manager = ResourceManager()

def random_uuid() -> str:
    return uuid.uuid4().hex


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk
            else:
                content = await response.read()
                yield content


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_data = await request.get_json()
        prefill_data = original_data.copy()
        prefill_data['max_tokens'] = 1

        prefill_addr, prefill_zmq = resource_manager.get_random("P")
        decode_addr, decode_zmq = resource_manager.get_random("D")
        print(f"handle_request, prefill: {prefill_addr}/{prefill_zmq}, decode: {decode_addr}/{decode_zmq}")

        request_id = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"

        # Prefill stage
        async for _ in forward_request(
                f"http://{prefill_addr}/v1/completions", prefill_data, request_id):
            pass

        # Decode stage and return
        generator = forward_request(
            f"http://{decode_addr}/v1/completions", original_data, request_id)
        response = await make_response(generator)
        response.timeout = None
        return response

    except Exception as e:
        import traceback
        print("Error in disagg proxy:", e)
        traceback.print_exc()


if __name__ == "__main__":
    start_service_discovery("0.0.0.0", 30001, resource_manager)
    app.run(host='0.0.0.0', port=10001)
