# SPDX-License-Identifier: Apache-2.0
# reference https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

# === export types and functions from flagcx to Python ===
# for the original flagcx definition, please check
# https://github.com/FlagOpen/FlagCX/blob/main/flagcx/include/flagcx.h

flagcxResult_t = ctypes.c_int
flagcxDataType_t = ctypes.c_int
flagcxRedOp_t = ctypes.c_int
flagcxMemcpyType_t = ctypes.c_int
flagcxMemType_t = ctypes.c_int

flagcxHandlerGroup_t = ctypes.c_void_p
flagcxComm_t = ctypes.c_void_p
flagcxEvent_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p


class flagcxStream(ctypes.Structure):
    _fields_ = [("base", cudaStream_t)]
flagcxStream_t = ctypes.POINTER(flagcxStream)


class flagcxUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 256)]
flagcxUniqueId_t = ctypes.POINTER(flagcxUniqueId)


DEVICE_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t)
DEVICE_MEMCPY_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    flagcxMemcpyType_t, flagcxStream_t
)
DEVICE_MEMSET_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    flagcxMemType_t, flagcxStream_t
)
DEVICE_MALLOC_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
    flagcxMemType_t, flagcxStream_t
)
DEVICE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(
    flagcxResult_t, ctypes.c_void_p, flagcxMemType_t, flagcxStream_t
)
SET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_int)
GET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_DEVICE_COUNT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(ctypes.c_int))
GET_VENDOR_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.c_char_p)

STREAM_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxStream_t))
STREAM_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_COPY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxStream_t), ctypes.c_void_p)
STREAM_FREE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t)
STREAM_WAIT_EVENT_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxStream_t, flagcxEvent_t)

EVENT_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, ctypes.POINTER(flagcxEvent_t))
EVENT_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_RECORD_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t, flagcxStream_t)
EVENT_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
EVENT_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(flagcxResult_t, flagcxEvent_t)
class flagcxDeviceHandle(ctypes.Structure):
    _fields_ = [
        # Basic functions
        ("deviceSynchronize", DEVICE_SYNCHRONIZE_FUNCTYPE),
        ("deviceMemcpy", DEVICE_MEMCPY_FUNCTYPE),
        ("deviceMemset", DEVICE_MEMSET_FUNCTYPE),
        ("deviceMalloc", DEVICE_MALLOC_FUNCTYPE),
        ("deviceFree", DEVICE_FREE_FUNCTYPE),
        ("setDevice", SET_DEVICE_FUNCTYPE),
        ("getDevice", GET_DEVICE_FUNCTYPE),
        ("getDeviceCount", GET_DEVICE_COUNT_FUNCTYPE),
        ("getVendor", GET_VENDOR_FUNCTYPE),
        # Stream functions
        ("streamCreate", STREAM_CREATE_FUNCTYPE),
        ("streamDestroy", STREAM_DESTROY_FUNCTYPE),
        ("streamCopy", STREAM_COPY_FUNCTYPE),
        ("streamFree", STREAM_FREE_FUNCTYPE),
        ("streamSynchronize", STREAM_SYNCHRONIZE_FUNCTYPE),
        ("streamQuery", STREAM_QUERY_FUNCTYPE),
        ("streamWaitEvent", STREAM_WAIT_EVENT_FUNCTYPE),
        # Event functions
        ("eventCreate", EVENT_CREATE_FUNCTYPE),
        ("eventDestroy", EVENT_DESTROY_FUNCTYPE),
        ("eventRecord", EVENT_RECORD_FUNCTYPE),
        ("eventSynchronize", EVENT_SYNCHRONIZE_FUNCTYPE),
        ("eventQuery", EVENT_QUERY_FUNCTYPE),
    ]
flagcxDeviceHandle_t = ctypes.POINTER(flagcxDeviceHandle)

class flagcxHandlerGroup(ctypes.Structure):
    _fields_ = [
        ("uniqueId", flagcxUniqueId_t),
        ("comm", flagcxComm_t),
        ("devHandle", flagcxDeviceHandle_t),
    ]
flagcxHandlerGroup_t = ctypes.POINTER(flagcxHandlerGroup)


class flagcxDataTypeEnum:
    flagcxInt8 = 0
    flagcxChar = 0
    flagcxUint8 = 1
    flagcxInt32 = 2
    flagcxInt = 2
    flagcxUint32 = 3
    flagcxInt64 = 4
    flagcxUint64 = 5
    flagcxFloat16 = 6
    flagcxHalf = 6
    flagcxFloat32 = 7
    flagcxFloat = 7
    flagcxFloat64 = 8
    flagcxDouble = 8
    flagcxBfloat16 = 9
    flagcxNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.flagcxInt8
        if dtype == torch.uint8:
            return cls.flagcxUint8
        if dtype == torch.int32:
            return cls.flagcxInt32
        if dtype == torch.int64:
            return cls.flagcxInt64
        if dtype == torch.float16:
            return cls.flagcxFloat16
        if dtype == torch.float32:
            return cls.flagcxFloat32
        if dtype == torch.float64:
            return cls.flagcxFloat64
        if dtype == torch.bfloat16:
            return cls.flagcxBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


class flagcxRedOpTypeEnum:
    flagcxSum = 0
    flagcxProd = 1
    flagcxMax = 2
    flagcxMin = 3
    flagcxAvg = 4
    flagcxNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.flagcxSum
        if op == ReduceOp.PRODUCT:
            return cls.flagcxProd
        if op == ReduceOp.MAX:
            return cls.flagcxMax
        if op == ReduceOp.MIN:
            return cls.flagcxMin
        if op == ReduceOp.AVG:
            return cls.flagcxAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class FLAGCXLibrary:
    exported_functions = [
        Function("flagcxHandleInit", flagcxResult_t,
                [ctypes.POINTER(flagcxHandlerGroup_t)]),
        Function("flagcxHandleFree", flagcxResult_t,
                [flagcxHandlerGroup_t]),
        Function("flagcxGetErrorString", ctypes.c_char_p, [flagcxResult_t]),
        Function("flagcxGetVersion", flagcxResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        Function("flagcxGetUniqueId", flagcxResult_t,
                [ctypes.POINTER(ctypes.POINTER(flagcxUniqueId))]),
        # Note that flagcxComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("flagcxCommInitRank", flagcxResult_t, [
            ctypes.POINTER(flagcxComm_t), ctypes.c_int, ctypes.POINTER(flagcxUniqueId),
            ctypes.c_int
        ]),
        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxAllReduce", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxRedOp_t, flagcxComm_t, flagcxStream_t
        ]),

        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxAllGather", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxComm_t, flagcxStream_t
        ]),

        # Note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxReduceScatter", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxRedOp_t, flagcxComm_t, flagcxStream_t
        ]),

        Function("flagcxSend", flagcxResult_t, [
            buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int,
            flagcxComm_t, flagcxStream_t
        ]),

        Function("flagcxRecv", flagcxResult_t, [
            buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int,
            flagcxComm_t, flagcxStream_t
        ]),

        Function("flagcxBroadcast", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            ctypes.c_int, flagcxComm_t, flagcxStream_t
        ]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # flagcxResult_t flagcxCommDestroy(flagcxComm_t comm);
        Function("flagcxCommDestroy", flagcxResult_t, [flagcxComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):


        try:
            if so_file not in FLAGCXLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                FLAGCXLibrary.path_to_library_cache[so_file] = lib
            self.lib = FLAGCXLibrary.path_to_library_cache[so_file]
        except Exception as e:
            raise e

        if so_file not in FLAGCXLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in FLAGCXLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            FLAGCXLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = FLAGCXLibrary.path_to_dict_mapping[so_file]

        # init flagcx handler to call device-related apis
        self.handler = flagcxHandlerGroup_t()
        self.FLAGCX_CHECK(self._funcs["flagcxHandleInit"](ctypes.byref(self.handler)))

    def __del__(self):
        # free flagcx handler
        self.FLAGCX_CHECK(self._funcs["flagcxHandleFree"](self.handler))

    def flagcxGetErrorString(self, result: flagcxResult_t) -> str:
        return self._funcs["flagcxGetErrorString"](result).decode("utf-8")

    def FLAGCX_CHECK(self, result: flagcxResult_t) -> None:
        if result != 0:
            error_str = self.flagcxGetErrorString(result)
            raise RuntimeError(f"FLAGCX error: {error_str}")

    def flagcxGetVersion(self) -> str:
        version = ctypes.c_int()
        self.FLAGCX_CHECK(self._funcs["flagcxGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def flagcxGetUniqueId(self) -> flagcxUniqueId:
        unique_id = ctypes.POINTER(flagcxUniqueId)()
        self.FLAGCX_CHECK(self._funcs["flagcxGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> flagcxUniqueId:
        """
        Reconstructs an `ncclUniqueId` object from bytes data.
        Args:
            data: Must be a 128-byte data block (matching NCCL's unique_id).
        Returns:
            ncclUniqueId: The reconstructed NCCL Unique ID object.
        Raises:
            ValueError: If the input data length is not 128 bytes.
        """
        if len(data) != 256:
            raise ValueError(
                f"Expected 256 bytes for ncclUniqueId, got {len(data)} bytes")

        unique_id = flagcxUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 256)
        return unique_id

    def flagcxCommInitRank(self, world_size: int, unique_id: flagcxUniqueId,
                         rank: int) -> flagcxComm_t:
        comm = flagcxComm_t()
        self.FLAGCX_CHECK(self._funcs["flagcxCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def flagcxAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: flagcxComm_t,
                      stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # and `op` should be `flagcxRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def flagcxReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, op: int, comm: flagcxComm_t,
                          stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # and `op` should be `flagcxRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def flagcxAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: flagcxComm_t,
                      stream: flagcxStream_t) -> None:
        # `datatype` actually should be `flagcxDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.FLAGCX_CHECK(self._funcs["flagcxAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def flagcxSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def flagcxRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: flagcxComm_t, stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def flagcxBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, root: int, comm: flagcxComm_t,
                      stream: flagcxStream_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def flagcxCommDestroy(self, comm: flagcxComm_t) -> None:
        self.FLAGCX_CHECK(self._funcs["flagcxCommDestroy"](comm))

    def adaptor_stream_create(self):
        new_stream = flagcxStream_t()
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamCreate(ctypes.byref(new_stream)))
        return new_stream

    def adaptor_stream_copy(self, old_stream):
        new_stream = flagcxStream_t()
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamCopy(ctypes.byref(new_stream), ctypes.byref(cudaStream_t(old_stream.cuda_stream))))
        return new_stream

    def adaptor_stream_free(self, stream):
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamFree(stream))

    def adaptor_stream_destroy(self, stream):
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamDestroy(stream))

    def sync_stream(self, stream):
        self.FLAGCX_CHECK(self.handler.contents.devHandle.contents.streamSynchronize(stream))


__all__ = [
    "FLAGCXLibrary", "flagcxDataTypeEnum", "flagcxRedOpTypeEnum", "flagcxUniqueId",
    "flagcxHandlerGroup_t", "flagcxComm_t", "flagcxStream_t", "flagcxEvent_t", "buffer_type", "cudaStream_t"
]