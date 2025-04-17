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
# 日志配置
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ResourceManager: 统一管理 P/D 实例及其负载
# -----------------------------------------------------------------------------
class ResourceManager:
    def __init__(self):
        self._lock = threading.Lock()
        # 每个资源类型 'P' 或 'D' 映射到 {http_addr: {'zmq': zmq_addr, 'load': int}}
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
                # 如果 zmq 地址更新，则同步
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
# 全局对象与配置
# -----------------------------------------------------------------------------
rm = ResourceManager()

# 兼容旧版注册字典与 Condition，保留供外部等待
prefill_instances: dict[str, str] = {}
decode_instances: dict[str, str] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()

# 调度策略：random 或 least（最少负载）
SCHEDULING_STRATEGY = os.environ.get('SCHEDULING_STRATEGY', 'random').lower()

# -----------------------------------------------------------------------------
# 服务发现：接收实例注册
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
# HTTP 代理与请求转发
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

@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_data = await request.get_json()
        # 预填充请求：max_tokens=1
        prefill_request = original_data.copy()
        prefill_request['max_tokens'] = 1

        # 选择 Prefill 实例
        if SCHEDULING_STRATEGY == 'least':
            prefill_addr, prefill_zmq = rm.get_least_loaded('P')
        else:
            prefill_addr, prefill_zmq = rm.get_random('P')
        logger.info(f"Selected P-instance {prefill_addr} via '{SCHEDULING_STRATEGY}'")

        # 选择 Decode 实例
        if SCHEDULING_STRATEGY == 'least':
            decode_addr, decode_zmq = rm.get_least_loaded('D')
        else:
            decode_addr, decode_zmq = rm.get_random('D')
        logger.info(f"Selected D-instance {decode_addr} via '{SCHEDULING_STRATEGY}'")

        # 保持原始 request_id 组装格式
        request_id = (
            f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"
        )

        # 执行 Prefill，并更新负载
        rm.increment_load('P', prefill_addr)
        try:
            async for _ in forward_request(f'http://{prefill_addr}/v1/completions',
                                           prefill_request, request_id):
                pass
        finally:
            rm.decrement_load('P', prefill_addr)

        # 执行 Decode，并更新负载
        async def tracked_decode():
            rm.increment_load('D', decode_addr)
            try:
                async for chunk in forward_request(f'http://{decode_addr}/v1/completions',
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
# 启动
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    listener = start_service_discovery("0.0.0.0", 30001)
    app.run(host='0.0.0.0', port=10001)
    listener.join()
