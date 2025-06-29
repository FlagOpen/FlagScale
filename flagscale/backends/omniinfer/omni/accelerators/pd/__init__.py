from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory


def register():
    KVConnectorFactory.register_connector(
        "AscendHcclConnectorV1",
        "omni.accelerators.pd.llmdatadist_connector_v1",
        "LLMDataDistConnector"
    )