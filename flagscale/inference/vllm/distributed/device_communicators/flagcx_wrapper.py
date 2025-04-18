# SPDX-License-Identifier: Apache-2.0
# reference https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger

logger = init_logger(__name__)

# === export types and functions from flagcx to Python ===
# for the original flagcx definition, please check
# https://github.com/FlagOpen/FlagCX/blob/main/flagcx/include/flagcx.h

flagcxResult_t = ctypes.c_int
flagcxComm_t = ctypes.c_void_p

class flagcxUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 256)]

cudaStream_t = ctypes.c_void_p
flagcxStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

flagcxDataType_t = ctypes.c_int


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


flagcxRedOp_t = ctypes.c_int


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
        # const char *flagcxGetErrorString(flagcxResult_t result);
        Function("flagcxGetErrorString", ctypes.c_char_p, [flagcxResult_t]),
        # flagcxResult_t flagcxGetVersion(int *version);
        Function("flagcxGetVersion", flagcxResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # flagcxResult_t flagcxGetUniqueId(flagcxUniqueId_t *uniqueId);
        Function("flagcxGetUniqueId", flagcxResult_t,
                [ctypes.POINTER(ctypes.POINTER(flagcxUniqueId))]),
                #  [ctypes.POINTER(ctypes.POINTER(flagcxUniqueId))]),
        # flagcxResult_t flagcxCommInitRank(flagcxComm_t *comm, int nranks,
        #                           flagcxUniqueId_t commId, int rank);
        # note that flagcxComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("flagcxCommInitRank", flagcxResult_t, [
            ctypes.POINTER(flagcxComm_t), ctypes.c_int, ctypes.POINTER(flagcxUniqueId),
            ctypes.c_int
        ]),
        # flagcxResult_t flagcxAllReduce(const void *sendbuff, void *recvbuff,
        #                        size_t count, flagcxDataType_t datatype,
        #                        flagcxRedOp_t op, flagcxComm_t comm,
        #                        flagcxStream_t stream);
        # note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxAllReduce", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxRedOp_t, flagcxComm_t, flagcxStream_t
        ]),

        # flagcxResult_t flagcxAllGather(const void *sendbuff, void *recvbuff,
        #                        size_t sendcount, flagcxDataType_t datatype,
        #                        flagcxComm_t comm, flagcxStream_t stream);
        # note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxAllGather", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxComm_t, flagcxStream_t
        ]),

        # flagcxResult_t flagcxReduceScatter(const void *sendbuff, void *recvbuff,
        #                            size_t recvcount, flagcxDataType_t datatype,
        #                            flagcxRedOp_t op, flagcxComm_t comm,
        #                            flagcxStream_t stream);
        # note that flagcxStream_t is a pointer type, so the last argument
        # is a pointer
        Function("flagcxReduceScatter", flagcxResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, flagcxDataType_t,
            flagcxRedOp_t, flagcxComm_t, flagcxStream_t
        ]),

        # flagcxResult_t flagcxSend(const void *sendbuff, size_t count,
        #                   flagcxDataType_t datatype, int peer,
        #                   flagcxComm_t comm, flagcxStream_t stream);
        Function("flagcxSend", flagcxResult_t, [
            buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int,
            flagcxComm_t, flagcxStream_t
        ]),

        # flagcxResult_t flagcxRecv(void *recvbuff, size_t count,
        #                   flagcxDataType_t datatype, int peer,
        #                   flagcxComm_t comm, flagcxStream_t stream);
        Function("flagcxRecv", flagcxResult_t, [
            buffer_type, ctypes.c_size_t, flagcxDataType_t, ctypes.c_int,
            flagcxComm_t, flagcxStream_t
        ]),

        # flagcxResult_t flagcxBroadcast(const void *sendbuff, void *recvbuff,
        #                        size_t count, flagcxDataType_t datatype,
        #                        int root, flagcxComm_t comm,
        #                        flagcxStream_t stream);
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

        # flagcxResult_t cudaAdaptorStreamCopy(flagcxStream_t *newStream,
        #                              void *oldStream)
        Function("_Z21cudaAdaptorStreamCopyPP12flagcxStreamPv", flagcxResult_t, [ctypes.POINTER(flagcxStream_t), flagcxStream_t]),

        # flagcxResult_t cudaAdaptorStreamFree(flagcxStream_t stream)
        Function("_Z21cudaAdaptorStreamFreeP12flagcxStream", flagcxResult_t, [flagcxStream_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file #or find_flagcx_library()

        try:
            if so_file not in FLAGCXLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                FLAGCXLibrary.path_to_library_cache[so_file] = lib
            self.lib = FLAGCXLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load flagCX library from %s. "
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the flagcx library might not exist, be corrupted "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct flagcx library path.", so_file,
                platform.platform())
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
    # def flagcxGetUniqueId(self, unique_id: flagcxUniqueId) -> flagcxUniqueId:
    #     # unique_id = ctypes.POINTER(flagcxUniqueId)()
    #     self.FLAGCX_CHECK(self._funcs["flagcxGetUniqueId"](
    #         ctypes.byref(unique_id)))
    #     return unique_id

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

    def adaptor_stream_copy(self, old_stream: cudaStream_t):
        new_stream = flagcxStream_t()

        self.FLAGCX_CHECK(self._funcs["_Z21cudaAdaptorStreamCopyPP12flagcxStreamPv"](ctypes.byref(new_stream), ctypes.byref(old_stream)))
        return new_stream

    def adaptor_stream_free(stream):
        self.FLAGCX_CHECK(self._funcs["_Z21cudaAdaptorStreamFreeP12flagcxStream"](stream))
        result = lib.cudaAdaptorStreamFree(stream)

__all__ = [
    "FLAGCXLibrary", "flagcxDataTypeEnum", "flagcxRedOpTypeEnum", "flagcxUniqueId",
    "flagcxComm_t", "flagcxStream_t", "buffer_type", "cudaStream_t"
]
