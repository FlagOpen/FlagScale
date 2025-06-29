// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef MOEWEIGHTS_H
#define MOEWEIGHTS_H

#include <semaphore.h>
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
#include "tensor.h"

class ExpertWeights {
    // 专家类， 包含多个权重信息
    public:
        // 构造函数
        ExpertWeights() {}
        ExpertWeights(int expert_id, std::vector<Tensor> weights)
            : expert_id_(expert_id), weights_(weights){
                total_size_ = 0;
                for (auto& weight : weights_){
                    total_size_ += weight.get_total_size();
                }

        }
        // 新增公共方法获取私有成员
        int get_expert_id() const { return expert_id_; }
        size_t get_total_size() const {return total_size_;}
        aclError to_host(char * host_ptr) const{
            aclError ret;
            if (host_ptr == nullptr) {
                throw std::runtime_error("Target memory (host_ptr) is null");
            }
            for (const auto& weight : weights_){
                size_t data_size = weight.get_total_size();
                ret = weight.to_host((void*) (host_ptr));
                if (ret != ACL_ERROR_NONE) {
                    throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
                }
                host_ptr += data_size;
            }
            return ret;
        };
        aclError to_device(char * host_ptr){
            aclError ret;
            if (host_ptr == nullptr) {
                throw std::runtime_error("Target memory (host_ptr) is null");
            }
            for (auto& weight : weights_){
                size_t data_size = weight.get_total_size();
                ret = weight.to_device((void*) (host_ptr));
                if (ret != ACL_ERROR_NONE) {
                    throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
                }
                host_ptr += data_size;
            }
            return ret;
        };
    private:
        int expert_id_;        // FIXME: 初始化时全局专家id，权重替换后并不会更新该值
        std::vector<Tensor> weights_;         //该专家的多个权重，包含 bias， weight等信息
        size_t total_size_;           // 该专家权重参数数量
};

struct CountData {
    std::atomic<int> completed_processes;
    std::atomic<int> init_flag; // 用于标记是否已初始化
};

class MoEWeights {
private:
    std::vector<std::vector<ExpertWeights>> npu_weights_;

    void* count_ptr_;        // 共享引用计数

    std::string shm_name_ = "/omni_moe_weights";  // Shared memory name
    void* shm_ptr_;  // Pointer to shared DRAM
    size_t shm_size_;  // Total size in bytes

    size_t world_size_; //总进程数，用于分析共享内存的拷贝是否全部完成
    size_t num_layers_;
    size_t num_experts_;
    // Initialize shared memory
    void init_shared_memory(size_t shm_size);
    void replicate_to_shared_memory();

    // 创建或附加共享内存
    void* create_or_attach_shmem(const std::string& name, size_t size);

public:
    MoEWeights(size_t num_experts);
    MoEWeights(size_t num_experts, size_t world_size);
    ~MoEWeights();

    void init_weights(const std::vector<std::vector<std::vector<Tensor>>>& npu_weights, const std::vector<std::vector<int>>& expert_ids);

    void replacement(size_t layer_idx, size_t src_global_expert_idx, size_t dst_local_expert_idx);
    std::vector<std::vector<ExpertWeights>> getNpuWeights() const {return npu_weights_;}
    size_t getNumLayers() const {return num_layers_;}
    size_t getNumExperts() const {return num_experts_;}
    void* getShmPtr() const {return shm_ptr_;}
    bool isShmInitialized() const;
    size_t getShmSize() const {return shm_size_;}
    std::string getShmName() const {return shm_name_;}
    void unittest_for_init_shared_memory(size_t shm_size) {init_shared_memory(shm_size);} // 仅供Unitest调用
};
#endif // MOEWEIGHTS_H

