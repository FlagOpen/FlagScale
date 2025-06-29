// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef EXPERT_ACTIVATION_H
#define EXPERT_ACTIVATION_H

#include <vector>
#include <chrono>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cstring>
#include <thread> // 用于 std::this_thread::sleep_for
#include <mutex>  // 用于 std::unique_lock 和 std::mutex
#include <acl/acl.h>
#include "tensor.h"
#include <sys/stat.h> // New include for POSIX functions
#include <sys/types.h> // New include for POSIX types
#include <fstream>
#include <sstream>

class ExpertActivation {
private:
    struct Activation {
        double timestamp; // Time of the activation in seconds
        int64_t count;        // Number of activations at this timestamp
        Activation() : timestamp(0.0), count(0) {};
        Activation(double ts, int c);
    };

    Activation activationArray[12]; // Array to store activations with timestamps
    const size_t maxActivations;             // Maximum number of activations (x)
    size_t startIdx;                         // Start index for circular buffer
    size_t currentSize;                      // Current number of elements in the array
    const double timeThreshold;              // Time threshold in seconds (y)
    double lastActivationTime;               // Timestamp of the last processed activation
    int64_t pendingCount;                        // Accumulated count for activations < y seconds
    int __padding__[4];

    // Get current time in seconds (for simulation, using a simple system clock)
    double getCurrentTime() const;

public:
    ExpertActivation() : maxActivations(12), startIdx(0), currentSize(0), timeThreshold(5),
    lastActivationTime(0.0), pendingCount(0) {
        // activationArray.resize(maxActivations, Activation(0.0, 0)); // Pre-allocate array
    };

    // Constructor: x = max activations, y = time threshold in seconds
    ExpertActivation(size_t x, double y);

    // Add a new activation with a given count (default = 1)
    void addActivation(int64_t count = 1);

    // Get the total sum of activations in the activation array plus pending activations
    int64_t getTotalActivationCount() const;

    // For debugging: Print the current state of the activation array
    void printState() const;
};


class ClusterActivation
{
private:
    Tensor npu_count_;
    size_t num_layers_;
    size_t num_deploy_experts_;
    int activation_window_size_;
    size_t world_size_;
    size_t rank_;
    void* total_count_ptr_;
    void* last_count_ptr_;
    std::thread thread_;            // 工作线程
    bool enable_dump_ = false;
    std::string dump_dir_ = ""; // Fixed: Removed the reference, initialized an empty string

    enum ThreadState{
        INIT,
        RUNNING,
        STOPPING,
        STOPPED
    } thread_state_ = ThreadState::INIT;


    // Activation共享内存描述符
    std::string act_shm_name_ = "/omni_moe_activations";
    ExpertActivation* act_shm_ptr_;
    size_t act_shm_size_;

    void* create_or_attach_shmem(const std::string& name, size_t size);
    void init_activation_shmem();
    bool is_enbale_dump() const {return enable_dump_;}

public:
    ClusterActivation(Tensor npu_count,size_t num_layers, size_t num_deploy_experts, int activation_window_size,size_t world_size, size_t rank);
    ~ClusterActivation();
    void collect_activation(size_t layer_idx, size_t deploy_expert_idx, int64_t count);
    int64_t getClusterTotalActivationCount(size_t layer_idx, size_t deploy_expert_idx);
    void print_activations();
    void setDumpDir(const std::string& dump_dir);
    void stopDump();
    void dumpActivationCounts(size_t dump_count, int64_t* total_count_ptr, int64_t* last_count_ptr);
    size_t get_num_layers() const { return num_layers_; }
    size_t get_num_deploy_experts() const { return num_deploy_experts_; }
    size_t get_rank() const { return rank_; }
    size_t get_world_size() const { return world_size_; }

    //For Unittest
    Tensor& get_npu_count() { return npu_count_; }
    void* get_total_count_ptr() { return total_count_ptr_; }
    void* get_last_count_ptr() { return last_count_ptr_; }

    // 线程控制相关操作
    void collect_wrapper();
    void start_thread();
    void stop_thread();
};

#endif // EXPERT_ACTIVATION_H