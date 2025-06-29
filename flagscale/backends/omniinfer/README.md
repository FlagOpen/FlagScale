# Omni_Infer: Inference Accelerators for Ascend NPU

Omni_Infer is a powerful suite of inference accelerators tailored for the Ascend NPU platform, fully compatible with vLLM, and designed to deliver high-performance, enterprise-grade inference with native support and a growing feature set.

## Key Features

- **Enterprise-Grade Low-Latency P/D Scheduling**: xPyD scheduling with scale-out support for large-scale, disaggregated PD deployments, ensuring minimal latency. Refer to [Global Proxy Design](omni/accelerators/sched/global_proxy/README.md) for details.
- **Request-Level Load Balancing**: Optimizes prefill and decode phases for maximum throughput and low latency across all sequence lengths.
- **Optimized MoE Expert Deployment**: Supports large-scale Mixture of Experts (MoE) models with EP144/EP288 configurations.
- **MoE Expert Load Balancing**: Features layer-wise, uneven redundancy and near real-time dynamic expert placement for efficient resource utilization. Refer to [OmniPlacement Design](omni/accelerators/placement/README.md) for details.
- **Advanced Attention Optimizations**: Tailored for LLM, MLLM, and MoE models, enhancing performance and scalability.

## Getting Started

(SECTION TO BE COMPLETED)
To integrate Omni_Infer into your project, refer to the [installation guide](link-to-installation-guide) and [documentation](link-to-docs) for detailed setup instructions and API references.

## Contributing

(SECTION TO BE COMPLETED)
We welcome contributions to enhance Omni_Infer! Please check our [contributing guidelines](link-to-contributing) and submit pull requests or issues via [GitHub Issues](link-to-issues).

## License

Omni_Infer is licensed under the [MIT License](LICENSE).
