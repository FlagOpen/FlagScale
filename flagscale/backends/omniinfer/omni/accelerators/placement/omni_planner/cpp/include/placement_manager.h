// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef PLACEMENT_H
#define PLACEMENT_H

#include <mutex>
#include <vector>
#include <string>
#include <thread>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <assert.h>
#include <atomic>
#include "expert_activation.h"
#include "moe_weights.h"
#include "placement_mapping.h"
#include "placement_optimizer.h"

class Placement {
private:
    MoEWeights* moe_weight_;
    ClusterActivation* activations_;
    PlacementMapping* mapping_;
    PlacementOptimizer* optimizer_;

    int num_layers_;
    int rank_;                        // global device id
    int world_size_;                  // global device number
    int num_experts_;                 // num of logic expert, e.g. 256
    int num_deploy_experts_;          // num of physic deploy expert, e.g. 256 + 16(redundant)
    int num_deploy_experts_per_rank_; // num of physic deploy expert per ran, e.g. 17 + 1(redundant)
    int num_devices_per_host_;

    std::thread worker_thread_;    // Main worker thread
    std::thread init_thread_;      // Initialization thread
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> should_stop_init_{false};; // 新增，用于 init_thread_
    std::mutex mtx_;

public:
    Placement()
    : moe_weight_(nullptr),
    activations_(nullptr),
    mapping_(nullptr),
    optimizer_(nullptr),
    num_layers_(0),
    rank_(0),
    world_size_(0),
    num_experts_(0),
    num_deploy_experts_(0),
    num_deploy_experts_per_rank_(0),
    num_devices_per_host_(0){
    }
    Placement(int rank, int world_size, int num_devices_per_host, ClusterActivation* activation,
        size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
        size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype);

    ~Placement();

    void initialize_components(size_t expert_mapping_ptr, std::vector<int64_t> shape, int dtype,
        size_t placement_pattern_ptr, std::vector<int64_t> placement_shape, int placement_dtype);
    void check_shm_weights();
    void placement_manager();
    void replace_expert(int layer_id);

    // Thread control related operations
    void start_thread();
    void stop_thread();

    // UT test: Getter methods for private members
    MoEWeights* get_moe_weight() const { return moe_weight_; }
    ClusterActivation* get_activations() const { return activations_; }
    PlacementMapping* get_mapping() const { return mapping_; }
    PlacementOptimizer* get_optimizer() const { return optimizer_; }

    int get_num_layers() const { return num_layers_; }
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    int get_num_experts() const { return num_experts_; }
    int get_num_deploy_experts() const { return num_deploy_experts_; }
    int get_num_deploy_experts_per_rank() const { return num_deploy_experts_per_rank_; }
    int get_num_devices_per_host() const { return num_devices_per_host_; }

    std::thread& get_worker_thread() { return worker_thread_; }  // 返回引用，因为 std::thread 不可拷贝
    std::thread& get_init_thread() { return init_thread_; }      // 返回引用，因为 std::thread 不可拷贝
    bool get_should_stop() const { return should_stop_.load(); } // std::atomic 需要用 load() 获取值
    MoEWeights* get_moe_weights() {return moe_weight_ ;}
};

#endif // PLACEMENT_H