// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "expert_activation.h"
#include <iostream>
#include <thread>
#include "config.h"

namespace py = pybind11;

ExpertActivation::Activation::Activation(double ts, int c) : timestamp(ts), count(c) {}
ExpertActivation::ExpertActivation(size_t x, double y)
    : maxActivations(x), startIdx(0), currentSize(0), timeThreshold(y),
      lastActivationTime(0.0), pendingCount(0) {
    // activationArray.resize(maxActivations, Activation(0.0, 0)); // Pre-allocate array
}

double ExpertActivation::getCurrentTime() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

void ExpertActivation::addActivation(int64_t count) {
    double currentTime = getCurrentTime();

    // If this is the first activation or enough time has passed (>= y seconds)
    if (lastActivationTime == 0.0 || (currentTime - lastActivationTime) >= timeThreshold) {
        // If there are pending activations, add them to the array first
        if (pendingCount > 0) {
            // Calculate the next index (circular buffer)
            size_t nextIdx = (startIdx + currentSize) % maxActivations;

            // Add the pending activation
            //std::cout << "append new Activation("<<lastActivationTime << "," << pendingCount << ")" << std::endl;
            //std::cout << "Activation Array (size: " << currentSize << "), maxActivations (size: " << maxActivations << "), nextIdx (size: " << nextIdx << "):\n";
            if (nextIdx >= 12) {
                std::cout << "nextIdx " << nextIdx << " out of bounds" << std::endl;
                pendingCount += count;
                return;
            }
            activationArray[nextIdx] = Activation(lastActivationTime, pendingCount);

            // Update size or shift startIdx if array is full
            if (currentSize < maxActivations) {
                currentSize++;
            } else {
                startIdx = (startIdx + 1) % maxActivations;
            }
        }
        // Reset the pending count and update the last activation time
        pendingCount = count;
        lastActivationTime = currentTime;
    } else {
        // If less than y seconds, accumulate the count
        pendingCount += count;
    }
}

int64_t ExpertActivation::getTotalActivationCount() const {
    int64_t total = pendingCount; // Include pending activations
    for (size_t i = 0; i < currentSize; ++i) {
        size_t idx = (startIdx + i) % maxActivations;
        total += activationArray[idx].count;
    }
    return total;
}

void ExpertActivation::printState() const {
    std::cout << "Pending Count: " << pendingCount << " (Time: " << lastActivationTime << ")\n";
    std::cout << "Activation Array (size: " << currentSize << "):\n";
    for (size_t i = 0; i < currentSize; ++i) {
        size_t idx = (startIdx + i) % maxActivations;
        std::cout << "  Time: " << activationArray[idx].timestamp
                  << ", Count: " << activationArray[idx].count << "\n";
    }
}

void ClusterActivation::stopDump(){
    enable_dump_ = false;
}

void ClusterActivation::setDumpDir(const std::string& dump_dir) {
    struct stat info;
    if (stat(dump_dir.c_str(), &info) != 0) {
        if (mkdir(dump_dir.c_str(), 0755) != 0) {
            perror("Error creating directory");
            dump_dir_ = "";
            enable_dump_ = false;
        }
        else {
            dump_dir_ = dump_dir;
            enable_dump_ = true;
        }
    }
    else if (info.st_mode & S_IFDIR) {
        dump_dir_ = dump_dir;
        enable_dump_ = true;
    }
    else {
        dump_dir_ = "";
        enable_dump_ = false;
    }
}

ClusterActivation::ClusterActivation(Tensor npu_count,size_t num_layers, size_t num_deploy_experts, int activation_window_size,size_t world_size, size_t rank)
    : npu_count_(npu_count),num_layers_(num_layers), num_deploy_experts_(num_deploy_experts), activation_window_size_(activation_window_size), world_size_(world_size),rank_(rank)
{
    if (npu_count_.get_data_ptr()==nullptr)
    {
        throw std::invalid_argument("Current Tensor data_ptr() is nullptr!");
    }
    // 约束Tensor的 element_size 为 int
    if (npu_count_.get_element_size()!=sizeof(int64_t)){
        throw std::invalid_argument("Current Each Count Tensor Element Size is: "+std::to_string(npu_count_.get_element_size())+", while only support element size: "+ std::to_string(sizeof(int64_t))+" now");
    }
    if (get_rank()>=get_world_size()) {
        throw std::runtime_error("Current Rank is: "+std::to_string(get_rank())+" Current world_size is :"+ std::to_string(get_world_size()));
    }

    // Since local tokens global experts -> glocal tokens local experts, npu_coun_.get_length() must be less than get_num_layers()*(size_t)get_num_deploy_experts()
    if (npu_count_.get_length() > get_num_layers() * (size_t)get_num_deploy_experts()) {
        throw std::runtime_error("npu_count's length is: "+std::to_string(npu_count_.get_length())+" , which is larger than Current total layer and experts_per_layer is :"+ std::to_string(get_num_layers()) +"*"+std::to_string(get_num_deploy_experts()));
    }
    init_activation_shmem();

    size_t total_size = npu_count_.get_total_size(); // 用于分配Host内存
    total_count_ptr_ = malloc(total_size);
    memset(total_count_ptr_, 0, total_size);

    last_count_ptr_ = malloc(total_size);
    memset(last_count_ptr_, 0, total_size);

    thread_state_ = ThreadState::INIT;

    // 启动线程监听并更新专家激活信息
    start_thread();
}

ClusterActivation::~ClusterActivation()
{
    stop_thread(); // 析构时安全关闭线程
    if (act_shm_ptr_) {
        munmap(act_shm_ptr_, act_shm_size_);
        shm_unlink(act_shm_name_.c_str());
    }
    free(total_count_ptr_);
    free(last_count_ptr_);
}

// 创建或附加共享内存
void* ClusterActivation::create_or_attach_shmem(const std::string& name, size_t size) {
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) throw std::runtime_error("shm_open failed");

    if (ftruncate(fd, size) == -1) {
        close(fd);
        throw std::runtime_error("ftruncate failed");
    }

    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
    return ptr;
}


// 初始化激活共享内存
void ClusterActivation::init_activation_shmem() {
    act_shm_size_ = world_size_*num_layers_ * num_deploy_experts_ * sizeof(ExpertActivation);
    act_shm_ptr_ = static_cast<ExpertActivation*>(create_or_attach_shmem(act_shm_name_, act_shm_size_));
    // 同一个进程串行调用共享内存的初始化，会导致共享内存对象被反复初始化,
    // 改为只初始当前rank这块，
    for (size_t i =rank_*num_layers_ * num_deploy_experts_; i < (rank_+1)*num_layers_ * num_deploy_experts_; i++)
    {
        new (&act_shm_ptr_[i]) ExpertActivation();  // 使用placement new在已分配内存上构造对象
    }
}

void ClusterActivation::start_thread(){
    assert(thread_state_ == ThreadState::INIT);
    thread_state_ = ThreadState::RUNNING;
    thread_ = std::thread(&ClusterActivation::collect_wrapper, this);
}

void ClusterActivation::stop_thread() {
    thread_state_ = ThreadState::STOPPED;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ClusterActivation::dumpActivationCounts(size_t dump_count, int64_t* total_count_ptr, int64_t* last_count_ptr) {
    if (dump_dir_.empty()) {
        std::cerr << "Dump directory not set. Use setDumpDir first." << std::endl;
        return;
    }

    std::string filename = dump_dir_ + "/activation_counts_recordstep_" + std::to_string(dump_count) + "_rank_"+std::to_string(get_rank())+".txt";
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    std::ostringstream oss;
    for (size_t i = 0; i < num_layers_; ++i) {
        for (size_t j = 0; j < num_deploy_experts_; ++j) {
            size_t index = i * num_deploy_experts_ + j;
            int64_t countDiff = total_count_ptr[index] - last_count_ptr[index];
            oss << countDiff << "\t";
        }
        oss << std::endl;
    }

    outFile << oss.str();
    outFile.close();
}

void ClusterActivation::collect_wrapper() {
    aclInit(NULL); // 初始化 ACL
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtSetCurrentContext(context);

    size_t dump_count = 0;
    while(thread_state_ == ThreadState::RUNNING) {
        aclError ret = npu_count_.to_host(total_count_ptr_);
        if (ret != ACL_ERROR_NONE) {
            throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
        }
        int64_t* total_count_ptr = static_cast<int64_t*>(total_count_ptr_);
        int64_t* last_count_ptr = static_cast<int64_t*>(last_count_ptr_);

        if (is_enbale_dump()){
            dump_count += 1;
            dumpActivationCounts(dump_count,total_count_ptr,last_count_ptr);
        }

        for (size_t layer = 0; layer < get_num_layers(); ++layer) {
            for (size_t expert = 0; expert < get_num_deploy_experts(); ++expert){
                size_t idx = layer*get_num_deploy_experts()+expert;
                int64_t count =total_count_ptr[idx] - last_count_ptr[idx];
                if (count>0){
                    collect_activation(layer,expert,count); //TODO: kww， 由于local tokens global experts -> global tokens + local experts 的改变， 导致动态信息收集的出错，还需要一个映射修改
                    last_count_ptr[idx] = total_count_ptr[idx];
                }
                else if (count<0){
                    throw std::runtime_error("npu count value is less than last time value: "+std::to_string(total_count_ptr[idx])+"/ "+ std::to_string(last_count_ptr[idx]) + " on layer: "+std::to_string(layer)+"/"+std::to_string(get_num_layers())+" local_deploy expert idx: "+ std::to_string(expert)+"/"+std::to_string(get_num_deploy_experts())+" global index is: "+ std::to_string(idx)+" rank: "+ std::to_string(rank_));
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(config.activation_quiesce)); // Run every 30s
    }
    thread_state_ = ThreadState::STOPPED;
}

void ClusterActivation::collect_activation(size_t layer_idx, size_t deploy_expert_idx, int64_t count) {
    if (layer_idx >= num_layers_ || deploy_expert_idx >= num_deploy_experts_) {
        throw std::runtime_error("Invalid layer or expert");
    }
    const int idx = rank_* num_layers_ * num_deploy_experts_ +  layer_idx * num_deploy_experts_ + deploy_expert_idx;
    ExpertActivation &activation = act_shm_ptr_[idx];
    activation.addActivation(count);
}

// Get activations
int64_t ClusterActivation::getClusterTotalActivationCount(size_t layer_idx, size_t deploy_expert_idx) {
    if (layer_idx >= num_layers_ || deploy_expert_idx >= num_deploy_experts_) {
        throw std::runtime_error("Invalid layer or expert");
    }

    size_t idx = 0;
    int64_t totalActivation = 0;
    for (size_t i = 0; i < world_size_; i++)
    {
        idx = i* num_layers_ * num_deploy_experts_ +  layer_idx * num_deploy_experts_ + deploy_expert_idx;

        ExpertActivation &activation = act_shm_ptr_[idx];
        totalActivation = totalActivation + activation.getTotalActivationCount();
    }

    return totalActivation;
}


// 打印线程不再直接访问成员变量
// FIXME: Plz Consider different rank
void ClusterActivation::print_activations() {

    std::this_thread::sleep_for(std::chrono::minutes(1));

    std::cout << "Activations: ";
    for (size_t i = 0; i < num_layers_ * num_deploy_experts_; ++i) {
        act_shm_ptr_[i].printState();
    }
    std::cout << std::endl;
}
