import os
import random
import socket
import threading
import uuid
import logging

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ResourceManager: unified management of P/D instances and their load
# -----------------------------------------------------------------------------
class ResourceManager:
    def __init__(self):
        self._lock = threading.Lock()
        # Each resource type 'P' or 'D' maps to {http_addr: {'zmq': zmq_addr, 'load': int}}
        self._instances: dict[str, dict[str, dict[str, object]]] = {
            'P': {},
            'D': {},
        }

    def register(self, rtype: str, http_addr: str, zmq_addr: str):
        with self._lock:
            if http_addr not in self._instances[rtype]:
                self._instances[rtype][http_addr] = {'zmq': zmq_addr, 'load': 0}
                logger.info(f"Registered new {rtype}-instance {http_addr} (zmq={zmq_addr})")
            else:
                # If zmq address changed, synchronize it
                self._instances[rtype][http_addr]['zmq'] = zmq_addr

    def increment_load(self, rtype: str, http_addr: str):
        with self._lock:
            self._instances[rtype][http_addr]['load'] += 1
            logger.debug(f"[{rtype}] +1 load on {http_addr}, now={self._instances[rtype][http_addr]['load']}")

    def decrement_load(self, rtype: str, http_addr: str):
        with self._lock:
            self._instances[rtype][http_addr]['load'] -= 1
            logger.debug(f"[{rtype}] -1 load on {http_addr}, now={self._instances[rtype][http_addr]['load']}")

    def get_random(self, rtype: str) -> tuple[str, str]:
        with self._lock:
            items = list(self._instances[rtype].items())
        http_addr, info = random.choice(items)
        return http_addr, info['zmq']

    def get_least_loaded(self, rtype: str) -> tuple[str, str]:
        with self._lock:
            http_addr, info = min(self._instances[rtype].items(),
                                  key=lambda kv: kv[1]['load'])
        return http_addr, info['zmq']

# -----------------------------------------------------------------------------
# Globals & configuration
# -----------------------------------------------------------------------------
rm = ResourceManager()

# Legacy registration dicts & Conditions retained for external waiting
prefill_instances: dict[str, str] = {}
decode_instances: dict[str, str] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()

# Scheduling strategy: 'random' or 'least' (least load)
SCHEDULING_STRATEGY = os.environ.get('SCHEDULING_STRATEGY', 'random').lower()

# -----------------------------------------------------------------------------
# Service discovery: receive instance registrations
# -----------------------------------------------------------------------------
def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_addr, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            typ = data.get("type")
            http_addr = data.get("http_address")
            zmq_addr = data.get("zmq_address")
            if typ == "P":
                with prefill_cv:
                    prefill_instances[http_addr] = zmq_addr
                rm.register('P', http_addr, zmq_addr)
            elif typ == "D":
                with decode_cv:
                    decode_instances[http_addr] = zmq_addr
                rm.register('D', http_addr, zmq_addr)
            else:
                logger.warning(f"Unexpected registration message: {data}")

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
        target=_listen_for_register,
        args=[poller, router_socket],
        daemon=True
    )
    listener.start()
    return listener

# -----------------------------------------------------------------------------
# HTTP proxy & request forwarding
# -----------------------------------------------------------------------------
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)

def random_uuid() -> str:
    return uuid.uuid4().hex

async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=url, json=data, headers=headers) as resp:
            if resp.status == 200:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk
            else:
                content = await resp.read()
                yield content

# support both /v1/completions and /v1/chat/completions
@app.route('/v1/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
async def handle_request():
    try:
        original_data = await request.get_json()
        endpoint = request.path  # this will be '/v1/completions' or '/v1/chat/completions'
        
        # Prefill request: max_tokens=1
        prefill_request = original_data.copy()
        prefill_request['max_tokens'] = 1

        # Select Prefill instance
        if SCHEDULING_STRATEGY == 'least':
            prefill_addr, prefill_zmq = rm.get_least_loaded('P')
        else:
            prefill_addr, prefill_zmq = rm.get_random('P')
        logger.info(f"Selected P-instance {prefill_addr} via '{SCHEDULING_STRATEGY}'")

        # Select Decode instance
        if SCHEDULING_STRATEGY == 'least':
            decode_addr, decode_zmq = rm.get_least_loaded('D')
        else:
            decode_addr, decode_zmq = rm.get_random('D')
        logger.info(f"Selected D-instance {decode_addr} via '{SCHEDULING_STRATEGY}'")

        # Keep original request_id composition format
        request_id = (
            f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"
        )

        # Execute Prefill and update load
        rm.increment_load('P', prefill_addr)
        try:
            async for _ in forward_request(f'http://{prefill_addr}{endpoint}',
                                           prefill_request, request_id):
                pass
        finally:
            rm.decrement_load('P', prefill_addr)

        # Execute Decode and update load
        async def tracked_decode():
            rm.increment_load('D', decode_addr)
            try:
                async for chunk in forward_request(f'http://{decode_addr}{endpoint}',
                                                   original_data, request_id):
                    yield chunk
            finally:
                rm.decrement_load('D', decode_addr)

        resp = await make_response(tracked_decode())
        resp.timeout = None
        return resp

    except Exception as e:
        logger.error("Error in proxy server", exc_info=e)
        return {"error": str(e)}, 500

# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    listener = start_service_discovery("0.0.0.0", 30001)
    app.run(host='0.0.0.0', port=10001)
    listener.join()
