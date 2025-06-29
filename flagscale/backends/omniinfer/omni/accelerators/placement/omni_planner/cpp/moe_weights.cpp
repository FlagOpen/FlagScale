// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>  // For shared memory (POSIX-like)
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <acl/acl.h>
#include <sys/file.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <utility>
#include "moe_weights.h"
#include <fstream>  // 用于文件操作



namespace py = pybind11;

MoEWeights::MoEWeights(size_t num_experts):num_experts_(num_experts){
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    world_size_ = 1;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

MoEWeights::MoEWeights(size_t num_experts, size_t world_size):world_size_(world_size),num_experts_(num_experts){
    shm_ptr_ = nullptr;
    count_ptr_ = nullptr;
    shm_unlink(shm_name_.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}


MoEWeights::~MoEWeights() {
    // 清理控制共享内存
    if (shm_ptr_) {
        munmap(shm_ptr_, shm_size_);
        munmap(count_ptr_, sizeof(CountData));
        shm_unlink(shm_name_.c_str());
    }
}

void MoEWeights::init_weights(const std::vector<std::vector<std::vector<Tensor>>>& npu_weights, const std::vector<std::vector<int>>& expert_ids) {

    if (npu_weights.size()== 0){
        throw std::runtime_error("npu_weights.size() is 0, which is the layer dimension");
    }
    if (npu_weights.size()!=expert_ids.size()){
        throw std::out_of_range("npu_weights.size() is not equals to expert_ids.size(), which is the layer dimension");
    }
    if (npu_weights[0].size() == 0){
        throw std::runtime_error("npu_weights[0].size() is 0, which is the experts dimension");
    }
    if (npu_weights[0].size()!=expert_ids[0].size()){
        throw std::out_of_range("npu_weights[0].size() is not equals to expert_ids[0].size(), which is the expert dimension");
    }

    npu_weights_.resize(npu_weights.size()); // 预分配层数
    for (size_t layer_idx = 0; layer_idx < npu_weights.size(); ++layer_idx) {
        std::vector<ExpertWeights> layer_experts;
        layer_experts.resize(npu_weights[layer_idx].size()); // 预分配专家数
        for (size_t expert_idx = 0; expert_idx < npu_weights[layer_idx].size(); ++expert_idx) {
            // 为每个专家创建 ExpertWeights 对象
            layer_experts.at(expert_idx) = ExpertWeights(expert_ids[layer_idx][expert_idx], npu_weights[layer_idx][expert_idx]);
            // layer_experts.emplace_back(expert_ids[layer_idx][expert_idx], npu_weights[layer_idx][expert_idx]);
        }
        npu_weights_.at(layer_idx) = std::move(layer_experts);
    }

    if (npu_weights_.size() == 0 || npu_weights_[0].size()== 0 ) {
        throw std::runtime_error("Invalid nums of layer or expert of npu_weights_: size cannot be 0");
    }
    num_layers_ = npu_weights_.size();
    size_t expert_size = npu_weights_[0][0].get_total_size(); // FIXEME: 每个专家的大小均一致
    if (expert_size == 0) {
        throw std::runtime_error("Invalid size: size cannot be 0");
    }

    size_t total_size = num_layers_ * num_experts_ * expert_size;
    init_shared_memory(total_size);
    replicate_to_shared_memory();   // Initial copy to shared memory

    // 拷贝完成计数
    CountData* count_ptr = static_cast<CountData*>(count_ptr_);
    count_ptr->completed_processes.fetch_add(1);
}


bool MoEWeights::isShmInitialized() const {
    if (count_ptr_==nullptr){
        // 还没有在共享内存完成初始化
        return false;
    }
    else{
        CountData* count_ptr = static_cast<CountData*>(count_ptr_);
        size_t count = count_ptr->completed_processes.load();
        return count== world_size_;
    }

}

// // 创建或附加共享内存
void* MoEWeights::create_or_attach_shmem(const std::string& name, size_t size) {
    int fd = shm_open(name.c_str(),O_CREAT | O_EXCL | O_RDWR, 0666);
    bool is_creator;
    if (fd >= 0) {
        is_creator = true;
        if (ftruncate(fd, shm_size_) == -1) {
            close(fd);
            shm_unlink(name.c_str());
            throw std::runtime_error("ftruncate failed");

        }
    } else if (errno == EEXIST) {
        // 共享内存已存在，直接打开
        fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            throw std::runtime_error("Failed to open existing shared memory");
        }
        is_creator = false;
    } else {
        throw std::runtime_error("shm_open failed");
    }

    size_t total_size = size + sizeof(CountData); // 加入计数符号位
    void* ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");

    count_ptr_ = ptr;
    ptr = static_cast<void*>(static_cast<char*>(ptr) + sizeof(CountData));


    CountData* count_ptr = static_cast<CountData*>(count_ptr_);
    if (is_creator){
        count_ptr->completed_processes.store(0);
        count_ptr->init_flag.store(999);
    }
    else{
        // 非创建 进程等待完成初始化操作
        size_t check_shm_unrelease = 0;
        while(true){
            // std::cout<< init_completed_count->completed_processes<<std::endl;
            if (count_ptr->init_flag.load() == 999){
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // 异常退出导致共享内存未被正常释放提示
            ++ check_shm_unrelease;
            if (check_shm_unrelease>20){
                std::cout<<"Noticed: A Long SHM Init time cost: " <<check_shm_unrelease <<"s , SHM seems unreleased, Pls Check /dev/shm"<<std::endl;
            }
        }
    }
    return ptr;
}

// Initialize shared memory
void MoEWeights::init_shared_memory(size_t shm_size) {
    assert(sizeof(CountData) <=64 && "sizeof(CountData) is larger than 64");
    shm_size_ = shm_size+64-sizeof(CountData);
    shm_ptr_ = static_cast<void*>(create_or_attach_shmem(shm_name_, shm_size_));
}


bool is_within_bounds(char* shm_ptr, size_t shm_size, char* shm_ptr_current, size_t cp_size) {
    // 检查指针有效性
    if (shm_ptr == nullptr || shm_ptr_current == nullptr) {
        return false;
    }

    // 计算当前偏移量
    ptrdiff_t offset = shm_ptr_current - shm_ptr;

    // 检查是否在起始地址之前
    if (offset < 0) {
        return false; // 当前指针在 shm_ptr 之前，无效
    }

    // 转换为无符号类型，避免负数问题
    size_t unsigned_offset = static_cast<size_t>(offset);

    // 检查是否超出总大小
    if (unsigned_offset + cp_size > shm_size) {
        return false; // 超出范围
    }

    // 检查溢出（可选，size_tmp 过大时）
    if (unsigned_offset + cp_size < unsigned_offset) {
        return false; // cp_size 太大导致溢出
    }

    return true; // 在范围内
}


// 1. Helper to copy weights to shared memory
void MoEWeights::replicate_to_shared_memory() {
    // 确保共享内存指针已初始化
    assert(shm_ptr_ != nullptr && "Shared memory pointer is not initialized");

    char* shm_ptr = static_cast<char*>(shm_ptr_);
    char* shm_ptr_current = nullptr;
    int layer_idx = -1;
    int expert_id = 0;

    // 遍历两层结构: num_layers -> experts_per_layer
    for (const auto& layer : npu_weights_) {
        layer_idx++;
        assert(layer.size() > 0);
        for (const auto& expert_weights : layer) {
            expert_id = expert_weights.get_expert_id();
            assert( expert_id>=0 && (size_t)expert_id<num_experts_);
            size_t expert_size = expert_weights.get_total_size();
            shm_ptr_current = shm_ptr + (layer_idx * num_experts_ + expert_id) * expert_size; // FIXME: 所有专家默认字节大小都一样

            // 检查共享内存地址拷贝范围是否合法
            if (not is_within_bounds(shm_ptr,shm_size_,shm_ptr_current,expert_size)){
                throw std::runtime_error("Target memory (shm_ptr_current) is unvalided!");
            }
            aclError ret = expert_weights.to_host(shm_ptr_current);
            if (ret != ACL_SUCCESS) {
                 throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
            }
            shm_ptr_current += expert_size;
        }
    }
}

/**
 * @brief 将共享内存中的专家权重替换到 Weights指定层的指定本地专家位置
 *
 * 该函数从共享内存 (shm_ptr_) 中根据全局专家索引 (src_global_expert_idx) 和层索引 (layer_idx)
 * 定位源数据，并将其拷贝到 NPU 权重数组 (npu_weights_) 中指定层(layer_idx)的本地专家位置 (dst_local_expert_idx)。
 *
 * @param layer_idx [in] 层索引，表示目标权重所在的层，必须小于 npu_weights_.size()
 * @param src_global_expert_idx [in] 全局专家索引，用于计算共享内存中源数据的偏移量
 * @param dst_local_expert_idx [in] 本地专家索引，表示当前层内的目标专家位置，必须小于该层的专家数
 * @throws std::runtime_error 如果 layer_idx 或 dst_local_expert_idx 超出有效范围
 * @note 假设共享内存中的数据按层和专家顺序连续存储，且每个专家的权重大小为 weights_size_per_expert_
 */
void MoEWeights::replacement(size_t layer_idx, size_t src_global_expert_id, size_t dst_local_expert_idx){
    // 将共享内存指针转换为 char* 以便按字节偏移
    char* shm_ptr = static_cast<char*>(shm_ptr_);
    char* shm_ptr_current = static_cast<char*>(shm_ptr_);

    // // 检查层索引是否有效
    if ( layer_idx >= npu_weights_.size() ) {
        throw std::runtime_error("Invalid layer_idx: " + std::to_string(layer_idx) +
                                     ", current weights only have " + std::to_string(npu_weights_.size()) + " layers");
    }

    // 检查目标本地专家索引是否有效
    if (src_global_expert_id >= num_experts_ ) {
        throw std::runtime_error("Invalid src_global_expert_id: " + std::to_string(src_global_expert_id) +
                                     ", max:  " + std::to_string(num_experts_-1));
    }

    // 获取当前层的所有专家权重
    auto& layer_experts = npu_weights_[layer_idx];

    // 检查目标本地专家索引是否有效
    if ( dst_local_expert_idx >= layer_experts.size() ) {
        throw std::runtime_error("Invalid dst_local_expert_idx: " + std::to_string(dst_local_expert_idx) +
                                     ", max:  " + std::to_string(layer_experts.size()-1));
    }

    ExpertWeights expert_weights = layer_experts[dst_local_expert_idx];

    size_t expert_size = expert_weights.get_total_size();
    shm_ptr_current += (layer_idx * num_experts_ + src_global_expert_id) * expert_size;

    // 检查共享内存地址拷贝范围是否合法
    if (not is_within_bounds(shm_ptr,shm_size_,shm_ptr_current,expert_size)){
        throw std::runtime_error("Target memory (shm_ptr_current) is unvalided!");
    }

    aclError ret = expert_weights.to_device(shm_ptr_current);

    if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
    }
}

// PYBIND11_MODULE(moe_weights, m) {
//     m.doc() = "MoE weights management with shared memory";

//     // 1. 先绑定 Weight 类
//     py::class_<Tensor>(m, "Tensor")
//         .def(py::init<void*, size_t, size_t, const std::string&>(), // 按实际构造函数参数补全
//             py::arg("data_ptr"),
//             py::arg("length"),
//             py::arg("element_size"),
//             py::arg("name"));

//     // 3. 绑定 MoEWeights 类
//     py::class_<MoEWeights>(m, "MoEWeights")
//         .def(py::init<size_t,size_t>(),py::arg("num_experts"),py::arg("world_size"))
//         .def("init_weights", &MoEWeights::init_weights,
//             py::arg("npu_weights"),py::arg("expert_ids"),
//             "Initialize with NPU weights");

//     m.doc() = "Expert activation management with shared memory";

// }