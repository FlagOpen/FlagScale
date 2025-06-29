// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "expert_activation.h"
#include "moe_weights.h"
// #include "placement_mapping.h"
#include "placement_optimizer.h"
#include "placement_manager.h"
#include "tensor.h"
#include "config.h"

OmniConfig config;

namespace py = pybind11;

/**
 * Constructor for Placement class
 *
 * @param rank Global device ID
 * @param world_size Number of devices in the world
 * @param num_devices_per_host Number of devices per host
 * @param activations Pointer to ClusterActivation object
 * @param expert_mapping_ptr Pointer to expert mapping data
 * @param shape Shape of expert mapping data
 * @param dtype Data type of expert mapping data
 * @param placement_pattern_ptr Pointer to placement pattern data
 * @param placement_shape Shape of placement pattern data
 * @param placement_dtype Data type of placement pattern data
 *
 * Calls initialize_components immediately and starts a separate thread to check shared memory weights
 */
Placement::Placement(int rank, int world_size, int num_devices_per_host, ClusterActivation* activations,
    size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
    size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype)
    : activations_(activations),
      rank_(rank),
      world_size_(world_size),
      num_devices_per_host_(num_devices_per_host) {

    // Initialize components immediately
    initialize_components(expert_mapping_ptr, shape, dtype,
                         placement_pattern_ptr, placement_shape, placement_dtype);

    // Start a separate thread to check shared memory weights
    init_thread_ = std::thread(&Placement::check_shm_weights, this);
    // init_thread_.detach();
}

void Placement::initialize_components(size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
    size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype) {

    assert(shape.size() == 2);
    int64_t expert_shape[2];
    memcpy(expert_shape, shape.data(), sizeof(int64_t) * 2);

    assert(placement_shape.size() == 3);
    int64_t place_shape[3];
    memcpy(place_shape, placement_shape.data(), sizeof(int64_t) * 3);
    mapping_ = new PlacementMapping(rank_, num_devices_per_host_, (int32_t *)expert_mapping_ptr, expert_shape, dtype,
                                    (int32_t *)placement_pattern_ptr, place_shape, placement_dtype);

    num_layers_ = mapping_->get_num_layers();
    num_experts_ = mapping_->get_num_experts();
    num_deploy_experts_ = mapping_->get_num_deploy_experts();
    num_deploy_experts_per_rank_ = num_deploy_experts_ / world_size_;

    moe_weight_ = new MoEWeights(num_experts_, world_size_);
    optimizer_ = new PlacementOptimizer(mapping_, activations_);
}

void Placement::check_shm_weights() {
    std::cout << "check_shm_weights start success." << std::endl;
    while (!should_stop_init_) { // 使用标志控制退出
        if (moe_weight_ && moe_weight_->isShmInitialized()) {
            start_thread();
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(config.activation_quiesce)); // Check every 30s
    }
}

Placement::~Placement() {
    stop_thread();
    delete moe_weight_;
    delete mapping_;
    delete optimizer_;
    // delete activations_;
}

// 等待合适的时机等待专家权重替换
void quiesce() {
    // wait 5s before move weights to new postion
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // TODO: triger by vLLM when token finish
}

/**
 *
 * This function is called by Placement::placement_manager to manage the placement of experts for a layer.
 * It takes a layer_id as input and performs the following tasks:
 * 1. Determine the expert to be replaced (source expert -> dst position)
 * 2. Change old expert postion to default position
 * 3. Wait 5s before move weights to new postion
 * 4. Move weights to new postion
 * 5. Change source expert postion to new postion
 * 6. Update mapping
 *
 * The output of this function is an array of three elements: src_expert_idx, dst_expert_position, and old_expert_idx.
 * The meaning of these elements is as follows:
 * - src_expert_idx: the idx of the source expert to be replaced
 * - dst_expert_position: the position where the source expert should be moved to
 * - old_expert_idx: the idx of the old expert that was previously at the dst position
 *
 * If the optimizer determines that no replacement is needed, the output will be (-1, -1, -1).
 */
void Placement::replace_expert(int layer_id) {
    // 1. optimizer determine the expert to be replaced (source expert -> dst position)
    auto [src_expert_idx, dst_expert_position, old_expert_idx] = optimizer_->optimize(layer_id);

    if (src_expert_idx == -1 || dst_expert_position == -1 || old_expert_idx == -1)
    {
        std::cout << "This period no need replace experts," << " src_expert_idx " << src_expert_idx << " dst_expert_position " << dst_expert_position << " old_expert_idx " << old_expert_idx << std::endl;
        return;
    }

    // 2. change old expert postion to default position
    int default_position = mapping_->get_default_mapping_position(layer_id, old_expert_idx);
    mapping_->change_pos_id(layer_id, old_expert_idx, default_position);

    quiesce();

    // 3. move weights to new postion
    if (dst_expert_position / num_deploy_experts_per_rank_ == rank_) {

        int local_pos = dst_expert_position % num_deploy_experts_per_rank_;

        moe_weight_->replacement(layer_id, src_expert_idx, local_pos);
    }


    // 4. change source expert postion to new postion
    mapping_->change_pos_id(layer_id, src_expert_idx, dst_expert_position);

    // 5. update mapping
    mapping_->update_Position_To_Expert_Mapping(layer_id, dst_expert_position, src_expert_idx);
    mapping_->update_Redundant_Expert_Mapping(layer_id, dst_expert_position, src_expert_idx);
}

void Placement::placement_manager() {
    aclInit(NULL); // 初始化 ACL
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtSetCurrentContext(context);

    std::cout << "placement worker thread started\n";
    while(!should_stop_) {
        for (int layer = 0; layer < num_layers_; ++layer) {
            replace_expert(layer);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1 * 60)); // Run every 1 mins
    }
    std::cout << "placement worker thread stoped\n";
}

void Placement::start_thread() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!worker_thread_.joinable()) {
        should_stop_ = false;
        worker_thread_ = std::thread(&Placement::placement_manager, this);
    }
}

void Placement::stop_thread() {
    std::lock_guard<std::mutex> lock(mtx_);
    should_stop_init_ = true; // 通知 init_thread_ 退出
    should_stop_ = true;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    if (init_thread_.joinable()) {
        init_thread_.join(); // 等待初始化线程完成
    }
}

PYBIND11_MODULE(omni_placement, m) {
    m.doc() = "MoE weights management with shared memory";

    // 绑定 set_ut_memcpy_fun 函数
    m.def("set_ut_memcpy_fun", &set_ut_memcpy_fun, "Set the UT memcpy function");

    // 绑定 unset_ut_memcpy_fun 函数
    m.def("unset_ut_memcpy_fun", &unset_ut_memcpy_fun, "Unset the UT memcpy function");

    // 1. 绑定 PlacementMapping 类
    py::class_<PlacementMapping>(m, "PlacementMapping")
        .def(py::init<const std::string& , int , int ,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>,
                        size_t , std::vector<int64_t>>(),
        py::arg("filename"), py::arg("rank"), py::arg("num_devices"),
        py::arg("rendundancy_mapping"), py::arg("rendundancy_mapping_shape"),
        py::arg("global_mapping"), py::arg("global_mapping_shape"),
        py::arg("count"), py::arg("count_shape"),
        py::arg("pattern"), py::arg("pattern_shape"));


    // 3. 绑定 MoEWeights 类
    py::class_<MoEWeights>(m, "MoEWeights")
        // 根据 moe_weights.h 文件修改了 构造函数的入参
        .def(py::init<size_t>(),py::arg("num_experts"))
        .def(py::init<size_t,size_t>(),py::arg("num_experts"),py::arg("world_size"))
        .def("isShmInitialized",&MoEWeights::isShmInitialized)
        .def("init_weights", &MoEWeights::init_weights,
            py::arg("npu_weights"),
            py::arg("expert_ids"),
            "Initialize with NPU weights");

    // 4. 绑定 Placement 类
    py::class_<Placement>(m, "Placement")
        .def(py::init<>())
        .def(py::init<int, int, int, ClusterActivation*, size_t, std::vector<int64_t>, int,
                        size_t, std::vector<int64_t>, int>(),
                py::arg("rank"),
                py::arg("world_size"),
                py::arg("num_devices_per_host"),
                py::arg("activation"),
                py::arg("expert_mapping_ptr"),
                py::arg("shape"),
                py::arg("dtype"),
                py::arg("placement_pattern_ptr"),
                py::arg("placement_shape"),
                py::arg("placement_dtype"))
        .def("get_moe_weights", &Placement::get_moe_weights, py::return_value_policy::reference);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<uint64_t, size_t, size_t, const std::string&>(), // 按实际构造函数参数补全
            py::arg("data_ptr"),
            py::arg("length"),
            py::arg("element_size"),
            py::arg("name"));

    py::class_<ClusterActivation>(m, "ClusterActivation")
        .def(py::init<Tensor,size_t, size_t, int, size_t, size_t>(), // 按实际构造函数参数补全
        py::arg("npu_count"),
        py::arg("layer"),
        py::arg("num_expert"),
        py::arg("window_size"),
        py::arg("world_size"),
        py::arg("rank"),
            "Initialize with expert activation")
        .def("collect_activation", &ClusterActivation::collect_activation,
            py::arg("layer"), py::arg("expert"), py::arg("count"),
            "Record an expert activation")
        .def("getClusterTotalActivationCount",&ClusterActivation::getClusterTotalActivationCount,
            py::arg("layer"), py::arg("expert"),""
        )
        .def("stop_thread",&ClusterActivation::stop_thread,"")
        .def("stopDump",&ClusterActivation::stopDump,"")
        .def("setDumpDir", &ClusterActivation::setDumpDir,
             py::arg("dump_dir"),
             "Set the dump path for the cluster activation")
        ;
}