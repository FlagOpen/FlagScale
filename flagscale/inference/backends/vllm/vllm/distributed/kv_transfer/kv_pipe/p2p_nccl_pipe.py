# SPDX-License-Identifier: Apache-2.0
"""P2P tensor transport based on FlagCX (NCCL‑compatible)
This file is a drop‑in replacement for the original *p2p_nccl_pipe.py* that
shipped with vLLM <https://github.com/FlagOpen/FlagScale>.
The only runtime dependency is ``flagcx_wrapper`` whose Python/ctypes wrapper
now returns *structs* (not pointers) from ``flagcxGetUniqueId`` and expects the
struct *by value* in ``flagcxCommInitRank``.  Make sure you have applied the
wrapper patch before using this module.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import typing
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import msgpack  # type: ignore
import torch
import zmq  # type: ignore

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.flagcx_wrapper import (
    FLAGCXLibrary,
    buffer_type,
    cudaStream_t,
    flagcxComm_t,
    flagcxDataTypeEnum,
)
from vllm.utils import current_stream, get_ip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _resolve_flagcx_library() -> str:
    """Return absolute path to *libflagcx.so*.

    Users can point ``FLAGCX_PATH`` to either the build directory or the
    installation root (containing *build/lib/libflagcx.so*).  A politely
    worded ``RuntimeError`` is raised when the library cannot be located.
    """

    env_path = os.getenv("FLAGCX_PATH")
    if not env_path:
        raise RuntimeError(
            "Environment variable FLAGCX_PATH is not set — cannot locate "
            "libflagcx.so.  Either install FlagCX system‑wide or export the "
            "build directory, e.g. \n    export FLAGCX_PATH=/path/to/FlagCX")

    # Accept both $FLAGCX_PATH/build/lib and $FLAGCX_PATH
    candidates = [
        os.path.join(env_path, "build/lib/libflagcx.so"),
        os.path.join(env_path, "libflagcx.so"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    raise RuntimeError(
        "Could not find libflagcx.so.  Checked:\n  • "
        + "\n  • ".join(candidates)
    )

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class P2pNcclPipe:
    """Point‑to‑point tensor pipe implemented with FlagCX."""

    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: Optional[str] = None,
    ) -> None:
        # ---------------------------- basic attributes ------------------------
        self.config = config
        self.rank = port_offset  # pipe‑level rank (0 for producer, 1 for consumer)
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")

        # ------------------------- load FlagCX library -----------------------
        self.flagcx = FLAGCXLibrary(library_path or _resolve_flagcx_library())

        # ------------------------- address bookkeeping -----------------------
        if not hostname:
            hostname = get_ip()
        self._hostname = hostname

        port = self.config.kv_port + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._port = port

        # Each GPU has its own ZMQ socket (ROUTER / DEALER pattern)
        self.zmq_address = f"{self._hostname}:{self._port}"
        self.http_address = (
            f"{self._hostname}:{self.config.kv_connector_extra_config['http_port']}"
        )

        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        self.proxy_address = f"{proxy_ip}:{proxy_port}" if proxy_ip and proxy_port else ""

        # -------------------- ZeroMQ context & sockets -----------------------
        self.context = zmq.Context.instance()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        # ------------------------------- state -------------------------------
        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self.comm_cv = threading.Condition()

        self.send_type = self.config.get_from_extra_config("send_type", "PUT")
        if self.send_type == "GET":
            self.send_store: Dict[str, torch.Tensor] = {}
        else:  # PUT or PUT_ASYNC
            self.send_queue: Deque[List[Any]] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self._send_async, daemon=True, name="p2p‑send‑async"
                )
                self._send_thread.start()

        self.recv_store: Dict[str, torch.Tensor] = {}
        self.socks: Dict[str, Any] = {}
        self.comms: Dict[str, Any] = {}

        self.buffer_size = 0
        self.buffer_size_threshold = self.config.kv_buffer_size

        # ------------------------- background threads ------------------------
        self._listener_thread = threading.Thread(
            target=self._listen_for_requests, daemon=True, name="p2p‑listener"
        )
        self._listener_thread.start()

        self._ping_thread: Optional[threading.Thread] = None
        if port_offset == 0 and self.proxy_address:
            self._ping_thread = threading.Thread(
                target=self._ping, daemon=True, name="p2p‑ping"
            )
            self._ping_thread.start()

    # ---------------------------------------------------------------------
    # Public API (send / recv)
    # ---------------------------------------------------------------------

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: typing.Optional[str] = None,
    ) -> bool:
        """Send *tensor* to *remote_address* (or loopback if ``None``)."""
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        if self.send_type == "PUT":
            return self._send_sync(tensor_id, tensor, remote_address)
        elif self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append([tensor_id, remote_address, tensor])
                self.send_queue_cv.notify()
            return True
        else:  # GET mode
            return self._enqueue_for_get(tensor_id, tensor)

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Blocking receive of the tensor *tensor_id* from *remote_address*."""
        if self.send_type in {"PUT", "PUT_ASYNC"}:
            return self._recv_blocking(tensor_id, remote_address)
        else:  # GET mode
            return self._recv_via_get(tensor_id, remote_address)

    # ---------------------------------------------------------------------
    # Internal helpers (connection management)
    # ---------------------------------------------------------------------

    def _create_connect(self, remote_address: str):
        if remote_address in self.socks:
            return self.socks[remote_address], self.comms[remote_address]

        # 1. ZMQ DEALER socket ---------------------------------------------
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.connect(f"tcp://{remote_address}")
        self.socks[remote_address] = sock

        # 2. FlagCX communicator -------------------------------------------
        #unique_id = self.flagcx.flagcxGetUniqueId()
        unique_id_ptr = self.flagcx.flagcxGetUniqueId()
        uid = unique_id_ptr.contents if hasattr(unique_id_ptr, "contents") else unique_id_ptr
        data = {"cmd": "NEW", "unique_id": bytes(uid.internal)}
#        data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
        sock.send(msgpack.dumps(data))

        with torch.cuda.device(self.device):
            rank = 0  # side that initiates the connection
            comm: flagcxComm_t = self.flagcx.flagcxCommInitRank(2, uid, rank) # cz unique_id --> uid
        self.comms[remote_address] = (comm, rank)
        logger.info("🤝 flagcxCommInitRank established %s → %s (rank=%d)",
                    self.zmq_address, remote_address, rank)
        return sock, self.comms[remote_address]

    # ---------------------------------------------------------------------
    # Internal helpers (send / recv implementations)
    # ---------------------------------------------------------------------

    def _send_sync(
        self, tensor_id: str, tensor: torch.Tensor, remote_address: str
    ) -> bool:
        sock, (comm, rank) = self._create_connect(remote_address)
        hdr = {
            "cmd": "PUT",
            "tensor_id": tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        sock.send(msgpack.dumps(hdr))
        if sock.recv() != b"0":
            logger.warning("Peer OOM/threshold hit — defer send of %s", tensor_id)
            return False

        self._flagcx_send(comm, tensor.to(self.device), rank ^ 1)
        logger.debug("Sent tensor %s → %s (%s)", tensor_id, remote_address, tensor.shape)
        return True

    def _send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                tensor_id, remote_address, tensor = self.send_queue.popleft()
            self._send_sync(tensor_id, tensor, remote_address)

    def _enqueue_for_get(self, tensor_id: str, tensor: torch.Tensor) -> bool:
        tensor_size = tensor.element_size() * tensor.numel()
        with self.send_store_cv:
            while (self.buffer_size + tensor_size) > self.buffer_size_threshold:
                # Drop LRU tensor
                old_id, old_tensor = self.send_store.popitem(last=False)
                self.buffer_size -= old_tensor.element_size() * old_tensor.numel()
                logger.debug("Dropped cached tensor %s to free space", old_id)
            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
        return True

    def _recv_blocking(
        self, tensor_id: str, remote_address: Optional[str]
    ) -> Optional[torch.Tensor]:
        start = time.time()
        with self.recv_store_cv:
            while tensor_id not in self.recv_store:
                self.recv_store_cv.wait()
            tensor = self.recv_store.pop(tensor_id)
        duration_ms = (time.time() - start) * 1e3
        logger.debug("Received tensor %s from %s in %.2f ms", tensor_id, remote_address, duration_ms)
        return tensor

    def _recv_via_get(
        self, tensor_id: str, remote_address: str
    ) -> Optional[torch.Tensor]:
        sock, (comm, rank) = self._create_connect(remote_address)
        sock.send(msgpack.dumps({"cmd": "GET", "tensor_id": tensor_id}))
        meta = msgpack.loads(sock.recv())
        if meta["ret"] != 0:
            logger.warning("Peer lacked tensor %s", tensor_id)
            return None

        tensor = torch.empty(meta["shape"], dtype=getattr(torch, meta["dtype"]), device=self.device)
        self._flagcx_recv(comm, tensor, rank ^ 1)
        return tensor

    # ---------------------------------------------------------------------
    # Background threads
    # ---------------------------------------------------------------------

    def _listen_for_requests(self):
        while True:
            for sock, _ in self.poller.poll():
                if sock is self.router_socket:
                    remote_address, payload = self.router_socket.recv_multipart()
                    self._handle_router_message(remote_address, payload)

    def _handle_router_message(self, remote_address: bytes, payload: bytes):
        data = msgpack.loads(payload)
        raddr = remote_address.decode()
        cmd = data["cmd"]

        if cmd == "NEW":
            unique_id = self.flagcx.unique_id_from_bytes(bytes(data["unique_id"]))
            with torch.cuda.device(self.device):
                rank = 1  # the *acceptor* side of the connection
                comm: flagcxComm_t = self.flagcx.flagcxCommInitRank(2, unique_id, rank)
            self.comms[raddr] = (comm, rank)
            logger.info("Established reverse connection %s ← %s", self.zmq_address, raddr)
            return

        if cmd == "PUT":
            self._handle_put(raddr, data)
        elif cmd == "GET":
            self._handle_get(raddr, data)
        else:
            logger.warning("Unknown command %s from %s", cmd, raddr)

    def _handle_put(self, raddr: str, data: dict[str, Any]):
        tensor = torch.empty(
            data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
        )
        tensor_size = tensor.element_size() * tensor.numel()
        if (self.buffer_size + tensor_size) > self.buffer_size_threshold:
            self.router_socket.send_multipart([raddr.encode(), b"2"])  # threshold hit
            return
        self.router_socket.send_multipart([raddr.encode(), b"0"])  # OK
        comm, rank = self.comms[raddr]
        self._flagcx_recv(comm, tensor, rank ^ 1)
        tensor_id = data["tensor_id"]
        with self.recv_store_cv:
            self.recv_store[tensor_id] = tensor
            self.recv_store_cv.notify()
        self.buffer_size += tensor_size

    def _handle_get(self, raddr: str, data: dict[str, Any]):
        tensor_id = data["tensor_id"]
        with self.send_store_cv:
            tensor = self.send_store.pop(tensor_id, None)
            if tensor is None:
                self.router_socket.send_multipart([raddr.encode(), msgpack.dumps({"ret": 1})])
                return
            meta = {
                "ret": 0,
                "shape": tensor.shape,
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
            self.router_socket.send_multipart([raddr.encode(), msgpack.dumps(meta)])
        comm, rank = self.comms[raddr]
        self._flagcx_send(comm, tensor.to(self.device), rank ^ 1)

    # ---------------------------------------------------------------------
    # Low‑level FlagCX wrappers (thread‑safe)
    # ---------------------------------------------------------------------

    def _flagcx_send(self, comm: flagcxComm_t, tensor: torch.Tensor, dst: int):
        assert tensor.device == self.device
        stream = current_stream()
        with self.comm_cv:
            self.flagcx.flagcxSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                flagcxDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )

    def _flagcx_recv(self, comm: flagcxComm_t, tensor: torch.Tensor, src: int):
        assert tensor.device == self.device
        stream = current_stream()
        with self.comm_cv:
            self.flagcx.flagcxRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                flagcxDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )

    # ---------------------------------------------------------------------
    # Ping for proxy registration
    # ---------------------------------------------------------------------

    def _ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        payload = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(payload))
            time.sleep(3)

    # ---------------------------------------------------------------------
    # Teardown
    # ---------------------------------------------------------------------

    def close(self) -> None:
        """Join background threads and close ZMQ context."""
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
        self.context.destroy(linger=0)