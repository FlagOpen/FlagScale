// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "placement_mapping.h"
#include <iostream>
#include <string.h>
#include <tuple>
#include <algorithm>  // for std::find
#include <assert.h>
#include <tensor.h>

const int ScalarType_Int = 3;  // WARNING: ?????????????????????????????

// PlacementMapping::PlacementMapping(int rank, int num_devices_per_host,
//     int32_t * expert_mapping_ptr, int64_t expert_shape[dimensions_of_expert_mapping_layers_experts], int shape_dtype,
//     int32_t * placement_pattern_ptr, int64_t placement_shape[dimensions_of_placement_pattern_devices_layers_experts], int placement_dtype,
//     bool enable_redundant_mapping)
//     : rank_(rank),
//     num_devices_per_host_(num_devices_per_host),
//     expert_mapping_(expert_mapping_ptr),
//     placement_pattern_(placement_pattern_ptr),
//     redundant_mapping_(nullptr)  // 初始为空
PlacementMapping::PlacementMapping(int rank, int num_devices_per_host,
    int32_t * expert_mapping_ptr, int64_t expert_shape[dimensions_of_expert_mapping_layers_experts], int shape_dtype,
    int32_t * placement_pattern_ptr, int64_t placement_shape[dimensions_of_placement_pattern_devices_layers_experts], int placement_dtype,
    const char* placement_pattern_filename,
    bool enable_redundant)
    : rank_(rank),
    num_devices_per_host_(num_devices_per_host),
    expert_mapping_(expert_mapping_ptr),
    redundant_mapping_(nullptr)
{
    memcpy(&expert_mapping_shape_, expert_shape, sizeof(int64_t) * dimensions_of_expert_mapping_layers_experts);
    expert_mapping_dtype_ = shape_dtype;

    // 检查是否提供了文件名且指针为空
    // bool use_file = (placement_pattern_filename && placement_pattern_filename[0] != '\0' && placement_pattern_ptr == nullptr);
    bool use_file = (placement_pattern_filename && placement_pattern_filename[0] != '\0');

    if (use_file) {
        // 从文件加载 placement_pattern
        placement_pattern_vector_ = load_placement_pattern_from_file(placement_pattern_filename);

        // 根据读取的数据计算参数
        world_size_ = placement_pattern_vector_.size();
        num_layers_ = placement_pattern_vector_[0].size();
        num_experts_ = placement_pattern_vector_[0][0].size();

        // 填充用于记录的形状数组
        placement_pattern_shape_[0] = world_size_;
        placement_pattern_shape_[1] = num_layers_;
        placement_pattern_shape_[2] = num_experts_;

        // placement_pattern_ 指针设为 nullptr，因为我们直接使用 vector
        placement_pattern_ = nullptr;
        placement_pattern_dtype_ = ScalarType_Int;  // 假设文件中的数据是 int32 类型
    } else {
        // 原来的方式：从内存指针构造
        if (placement_pattern_ptr == nullptr) {
            throw std::invalid_argument("Both placement_pattern_ptr and placement_pattern_filename are invalid");
        }

        memcpy(&placement_pattern_shape_, placement_shape, sizeof(int64_t) * dimensions_of_placement_pattern_devices_layers_experts);
        placement_pattern_dtype_ = placement_dtype;
        placement_pattern_ = placement_pattern_ptr;

        // 初始化基于 placement pattern 的属性
        world_size_ = placement_shape[0];
        num_layers_ = placement_shape[1];
        num_experts_ = placement_shape[2];

        // 转换 placement pattern tensor 为 3D vector
        placement_pattern_vector_ = torch_tensor_to_3d_vector_int32();
    }

    assert(world_size_ >= 1);
    assert(num_layers_ >= 1);
    assert(num_experts_ >= 1);

    // Calculate experts per device
    num_experts_per_device_ = num_experts_ / world_size_;

    assert(num_experts_per_device_ >= 1);

    // Calculate number of deployed experts
    int sum_deployed_experts = 0;
    for (size_t i = 0; i < placement_pattern_vector_[0][0].size(); i++) {
        sum_deployed_experts += placement_pattern_vector_[0][0][i];
    }

    num_deploy_experts_ = sum_deployed_experts * world_size_;
    num_deploy_experts_per_device_ = num_deploy_experts_ / world_size_;

    assert(num_deploy_experts_ >= num_experts_);
    assert(num_deploy_experts_per_device_ >= num_experts_per_device_);

    // Construct mappings
    position_to_epid_ = construct_position_mapping_to_epid();

    // 根据需要启用冗余专家映射
    if (enable_redundant) {
        this->enable_redundant_mapping();
    }
}

PlacementMapping::PlacementMapping(const std::string& placement_pattern_filename, int rank, int num_devices_per_host,
    size_t redundant_expert_mapping_ptr, std::vector<int64_t> redundant_mapping_shape,
    size_t global_expert_mapping_ptr, std::vector<int64_t> global_mapping_shape,
    size_t expert_redundant_count_ptr, std::vector<int64_t> count_shape,
    size_t placement_pattern_ptr, std::vector<int64_t> placement_shape)
    : rank_(rank),
    num_devices_per_host_(num_devices_per_host),
    redundant_expert_mapping_((int32_t *)redundant_expert_mapping_ptr),
    global_expert_mapping_((int32_t *)global_expert_mapping_ptr),
    redundant_count_per_expert_((int32_t *)expert_redundant_count_ptr)
{
    memcpy(&redundant_expert_mapping_shape_, redundant_mapping_shape.data(), sizeof(int64_t) * NUM_DIM_OF_MAPPING);
    memcpy(&global_expert_mapping_shape_, global_mapping_shape.data(), sizeof(int64_t) * NUM_DIM_OF_MAPPING);
    memcpy(&redundant_count_per_expert_shape_, count_shape.data(), sizeof(int64_t) * NUM_DIM_OF_COUNT);

    // 检查是否提供了文件名且指针为空
    bool use_file = (placement_pattern_filename.empty());

    if (false) {
        // 从文件加载 placement_pattern
        placement_pattern_vector_ = load_placement_pattern_from_file(placement_pattern_filename.c_str());

        // 根据读取的数据计算参数
        world_size_ = placement_pattern_vector_.size();
        num_layers_ = placement_pattern_vector_[0].size();
        num_experts_ = placement_pattern_vector_[0][0].size();

        // 填充用于记录的形状数组
        placement_pattern_shape_[0] = world_size_;
        placement_pattern_shape_[1] = num_layers_;
        placement_pattern_shape_[2] = num_experts_;

        // placement_pattern_ 指针设为 nullptr，因为我们直接使用 vector
        placement_pattern_ = nullptr;
        placement_pattern_dtype_ = ScalarType_Int;  // 假设文件中的数据是 int32 类型
    } else {
        // 原来的方式：从内存指针构造
        if (placement_pattern_ptr == 0) {
            throw std::invalid_argument("Both placement_pattern_ptr and placement_pattern_filename are invalid");
        }
        memcpy(&placement_pattern_shape_, placement_shape.data(), sizeof(int64_t) * NUM_DIM_OF_MAPPING);
        placement_pattern_dtype_ = 3;
        placement_pattern_ = (int32_t*)placement_pattern_ptr;

        // 初始化基于 placement pattern 的属性
        world_size_ = placement_shape[0];
        num_layers_ = placement_shape[1];
        num_experts_ = placement_shape[2];

        // 转换 placement pattern tensor 为 3D vector
        placement_pattern_vector_ = torch_tensor_to_3d_vector_int32();
    }

    assert(world_size_ >= 1);
    assert(num_layers_ >= 1);
    assert(num_experts_ >= 1);

    // Calculate experts per device
    num_experts_per_device_ = num_experts_ / world_size_;

    assert(num_experts_per_device_ >= 1);

    // Calculate number of deployed experts
    int sum_deployed_experts = 0;
    for (size_t i = 0; i < placement_pattern_vector_[0][0].size(); i++) {
        sum_deployed_experts += placement_pattern_vector_[0][0][i];
    }

    num_deploy_experts_ = sum_deployed_experts * world_size_;
    num_deploy_experts_per_device_ = num_deploy_experts_ / world_size_;

    assert(num_deploy_experts_ >= num_experts_);
    assert(num_deploy_experts_per_device_ >= num_experts_per_device_);

    // Construct mappings
    // position_to_epid_ = construct_position_mapping_to_epid();
    construct_epid_mapping_to_position_using_max_glabal_offset();
}


PlacementMapping::PlacementMapping(std::vector<std::vector<std::vector<int>>> &placement_pattern_vector, int rank, int num_devices_per_host,
    int32_t * redundant_expert_mapping_ptr, int64_t redundant_mapping_shape[NUM_DIM_OF_MAPPING],
    int32_t * global_expert_mapping_ptr, int64_t global_mapping_shape[NUM_DIM_OF_MAPPING],
    int32_t * expert_redundant_count_ptr, int64_t count_shape[NUM_DIM_OF_COUNT])
    : rank_(rank),
    num_devices_per_host_(num_devices_per_host),
    redundant_expert_mapping_(redundant_expert_mapping_ptr),
    global_expert_mapping_(global_expert_mapping_ptr),
    redundant_count_per_expert_(expert_redundant_count_ptr)
{
    memcpy(&redundant_expert_mapping_shape_, redundant_mapping_shape, sizeof(int64_t) * NUM_DIM_OF_MAPPING);
    memcpy(&global_expert_mapping_shape_, global_mapping_shape, sizeof(int64_t) * NUM_DIM_OF_MAPPING);
    memcpy(&redundant_count_per_expert_shape_, count_shape, sizeof(int64_t) * NUM_DIM_OF_COUNT);

    // 从文件加载 placement_pattern
    placement_pattern_vector_ = placement_pattern_vector;

    // 根据读取的数据计算参数
    world_size_ = placement_pattern_vector_.size();
    num_layers_ = placement_pattern_vector_[0].size();
    num_experts_ = placement_pattern_vector_[0][0].size();

    // 填充用于记录的形状数组
    placement_pattern_shape_[0] = world_size_;
    placement_pattern_shape_[1] = num_layers_;
    placement_pattern_shape_[2] = num_experts_;

    // placement_pattern_ 指针设为 nullptr，因为我们直接使用 vector
    placement_pattern_ = nullptr;
    placement_pattern_dtype_ = ScalarType_Int;  // 假设文件中的数据是 int32 类型

    assert(world_size_ >= 1);
    assert(num_layers_ >= 1);
    assert(num_experts_ >= 1);

    // Calculate experts per device
    num_experts_per_device_ = num_experts_ / world_size_;

    assert(num_experts_per_device_ >= 1);

    // Calculate number of deployed experts
    int sum_deployed_experts = 0;
    for (size_t i = 0; i < placement_pattern_vector_[0][0].size(); i++) {
        sum_deployed_experts += placement_pattern_vector_[0][0][i];
    }

    num_deploy_experts_ = sum_deployed_experts * world_size_;
    num_deploy_experts_per_device_ = num_deploy_experts_ / world_size_;

    assert(num_deploy_experts_ >= num_experts_);
    assert(num_deploy_experts_per_device_ >= num_experts_per_device_);

    // Construct mappings
    // position_to_epid_ = construct_position_mapping_to_epid();
    construct_epid_mapping_to_position_using_max_glabal_offset();
}

PlacementMapping::~PlacementMapping()
{
    // std::unique_ptr 会自动释放 redundant_mapping_ 资源
}

bool PlacementMapping::enable_redundant_mapping() {
    if (!redundant_mapping_) {
        try {
            redundant_mapping_ = std::make_unique<RedundantExpertMapping>(*this);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to enable redundant mapping: " << e.what() << std::endl;
            return false;
        }
    }
    return true; // 已经启用
}

void PlacementMapping::disable_redundant_mapping() {
    redundant_mapping_.reset();
}

// 从文件加载 placement pattern 的辅助方法
std::vector<std::vector<std::vector<int>>> PlacementMapping::load_placement_pattern_from_file(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        throw std::runtime_error(std::string("Failed to open placement pattern file: ") + filename);
    }

    // 读取形状 (3 个 int32_t 值)
    int32_t shape[3]; // num_devices * num_layers * num_experts
    if (fread(shape, sizeof(int32_t), 3, file) != 3) {
        fclose(file);
        throw std::runtime_error("Failed to read placement pattern shape from file");
    }

    // 根据形状初始化 3D vector
    std::vector<std::vector<std::vector<int>>> result(
        shape[0],
        std::vector<std::vector<int>>(
            shape[1],
            std::vector<int>(shape[2], 0)
        )
    );

    // 读取数据
    std::vector<int32_t> buffer(shape[0] * shape[1] * shape[2]);
    size_t read_elements = fread(buffer.data(), sizeof(int32_t), buffer.size(), file);
    fclose(file);

    if (read_elements != buffer.size()) {
        throw std::runtime_error("Failed to read placement pattern data from file");
    }

    // 将 1D 数组数据转换为 3D vector
    size_t idx = 0;
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            for (int k = 0; k < shape[2]; k++) {
                result[i][j][k] = buffer[idx++];

                // 验证数据的有效性
                assert(result[i][j][k] >= 0);
                assert(result[i][j][k] <= 1);
            }
        }
    }

    return result;
}

std::vector<std::vector<std::vector<int>>> PlacementMapping::torch_tensor_to_3d_vector_int32() {
    std::vector<std::vector<std::vector<int>>> result(world_size_, std::vector<std::vector<int>>(num_layers_, std::vector<int>(num_experts_)));

    if (placement_pattern_dtype_ == ScalarType_Int) {
        int32_t* data_ptr = placement_pattern_;
        for (int i = 0; i < world_size_; i++) {
            for (int j = 0; j < num_layers_; j++) {
                for (int k = 0; k < num_experts_; k++) {
                    int idx = i * (num_layers_ * num_experts_) + j * num_experts_ + k;
                    result[i][j][k] = data_ptr[idx];

                    assert(result[i][j][k] >= 0);
                    assert(result[i][j][k] <= 1);
                }
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type in torch_tensor_to_3d_vector_int32: " + std::to_string(placement_pattern_dtype_));
    }
    return result;
}

void PlacementMapping::change_pos_id(int layer_id, int expert_id, int new_position) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range");
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range");
    }
    if (new_position < 0 || new_position >= num_deploy_experts_) {
        throw std::out_of_range("new_position out of range");
    }

    int32_t offset = layer_id * num_experts_ + expert_id;

    // Get tensor data pointer and modify value at specific position
    int32_t* expert_data = static_cast<int32_t*>(expert_mapping_);
    aclError ret = (*get_memcpy_fun())(
        (void*)(expert_data + offset),  // Target address: base address + offset
        sizeof(int32_t),                // Target memory size
        &new_position,                  // Source data address
        sizeof(int32_t),                // Source data size
        ACL_MEMCPY_HOST_TO_DEVICE       // Transfer direction: host to device
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
    }
}

int32_t PlacementMapping::read_pos_id(int layer_id, int expert_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range");
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range");
    }

    int32_t offset = layer_id * num_experts_ + expert_id;
    int32_t value;
    aclError ret = (*get_memcpy_fun())(
        &value,
        sizeof(int32_t),
        static_cast<int32_t*>(expert_mapping_) + offset,
        sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " + std::to_string(ret));
    }
    return value;
}

int PlacementMapping::get_default_mapping_position(int layer_idx, int expert_id) {
    if (!redundant_mapping_) {
        throw std::runtime_error("Redundant expert mapping is not enabled");
    }
    return redundant_mapping_->get_default_mapping_position(layer_idx, expert_id);
}

std::vector<std::vector<int>> PlacementMapping::get_redundant_epid_this_host() {
    if (!redundant_mapping_) {
        return std::vector<std::vector<int>>();  // 返回空数组
    }
    return redundant_mapping_->get_redundant_epid_this_host();
}

std::vector<std::vector<int>> PlacementMapping::get_redundant_positionid_this_host() {
    if (!redundant_mapping_) {
        return std::vector<std::vector<int>>();  // 返回空数组
    }
    return redundant_mapping_->get_redundant_positionid_this_host();
}






void PlacementMapping::update_Redundant_Expert_Mapping(int layer_id, int target_position, int source_epid) {
    if (!redundant_mapping_) {
        throw std::runtime_error("Redundant expert mapping is not enabled");
    }
    redundant_mapping_->update_Redundant_Expert_Mapping(layer_id, target_position, source_epid);
}

std::vector<std::vector<int>> PlacementMapping::construct_position_mapping_to_epid() {
    std::vector<std::vector<int>> position_to_epid(num_layers_,
                                                   std::vector<int>(num_deploy_experts_, 0));

    for (int j = 0; j < num_layers_; j++) {
        int count = 0;
        for (int i = 0; i < world_size_; i++) {
            for (int k = 0; k < num_experts_; k++) {
                if (placement_pattern_vector_[i][j][k] == 1) {
                    position_to_epid[j][count] = k;
                    count++;

                    assert(count <= num_deploy_experts_);  // The position_mapping each layer should not exceed the  num_deploy_experts_.
                }
            }
        }
        assert(count == num_deploy_experts_); // The position_mapping part should contain all the num_deploy_experts_.
    }

    return position_to_epid;
}

std::vector<std::vector<int>> PlacementMapping::construct_placement_pattern_to_ep_mapping(
    const std::vector<std::vector<std::vector<int>>>& placement_pattern_here) {

    // Initialize expert_position with -1 (indicating no expert)
    std::vector<std::vector<int>> expert_position(
        num_layers_, std::vector<int>(num_experts_, -1));

    // Iterate through all devices, layers, and experts
    for (int j = 0; j < num_layers_; j++) {
        int count = 0;
        for (int i = 0; i < world_size_; i++) {
            for (int k = 0; k < num_experts_; k++) {
                if (placement_pattern_here[i][j][k] == 1) {
                    // Calculate sum of placement pattern up to k
                    int sum = 0;
                    for (int idx = 0; idx < k; idx++) {
                        sum += placement_pattern_vector_[i][j][idx];
                    }
                    // Calculate global expert index and fill into expert_position
                    expert_position[j][k] = i * num_deploy_experts_per_device_ + sum;

                    count++;
                    assert(count <= num_experts_); // This part should <= num_experts_
                }
            }
        }
        assert(count == num_experts_); // This part should contain all the num_experts_.
    }
    return expert_position;
}

void PlacementMapping::update_Position_To_Expert_Mapping(int layer_id, int target_position, int source_epid) {
    // 检查输入参数是否有效
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range");
    }
    if (target_position < 0 || target_position >= num_deploy_experts_) {
        throw std::out_of_range("target_position out of range");
    }
    if (source_epid < 0 || source_epid >= num_experts_) {
        throw std::out_of_range("source_epid out of range");
    }

    // 更新position_to_ep_mapping
    position_to_epid_[layer_id][target_position] = source_epid;
}

void PlacementMapping::construct_epid_mapping_to_position_using_max_glabal_offset() {


    // 声明每层每设备上部署的专家数量的最大值
    int32_t max_num_deployed_experts_device_each_layer = 0;

    // 第一阶段：计算所有层的所有设备里的最大加载数量
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        for (int32_t rank = 0; rank < world_size_; ++rank) {
            // 当前层当前设备部署的专家数量
            int32_t num_deployed_experts_device_this_layer = 0;

            // 统计当前层当前设备上部署的专家数量
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                if (placement_pattern_vector_[rank][layer][expert] == 1) {
                    num_deployed_experts_device_this_layer++;
                }
            }
            // 更新每层每设备专家部署数量的最大值
            max_num_deployed_experts_device_each_layer = std::max(max_num_deployed_experts_device_each_layer, num_deployed_experts_device_this_layer);
        }
    }

    // 第二阶段：使用全局最大偏移计算位置ID
    // 输入shape [num_layers_][num_experts_][max_redundant_count_]
    max_redundant_count_ = global_expert_mapping_shape_[2];

    // 使用所有层的所有设备里的最大加载数量作为offset，为所有层的所有设备计算position_id
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        for (int32_t rank = 0; rank < world_size_; ++rank) {
            // 用于累加当前设备上专家的位置
            int32_t position_id = 0;
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                // 如果当前专家在当前设备上部署
                if (placement_pattern_vector_[rank][layer][expert] == 1) {
                    // 获取当前专家的冗余计数
                    int redundant_count_this_expert = get_redundant_count(layer, expert);
                    // 确保冗余计数不超过最大允许值
                    assert(redundant_count_this_expert < max_redundant_count_);

                    // 计算全局位置ID：使用设备编号乘以每设备最大专家数作为基础偏移，再加上当前位置
                    int32_t position_id_here = max_num_deployed_experts_device_each_layer * rank + position_id;   // offset + cumsum;

                    // 设置专家的全局位置ID
                    set_global_expert_position_id(layer, expert, redundant_count_this_expert, position_id_here);
                    // 增加冗余计数
                    redundant_count_this_expert++;
                    // 更新冗余计数
                    set_redundant_count(layer, expert, redundant_count_this_expert);
                    // 增加位置计数器，为下一个专家做准备
                    position_id++; // for cumsum;
                }
            }
        }
    }

    // 为每次冗余构建独立的映射表
    construct_per_redundancy_epid_mapping_to_position();
}

// 为每次冗余构建独立的映射表
void PlacementMapping::construct_per_redundancy_epid_mapping_to_position() {
    // 维度：num_layers_ 、max_num_deployed_redundant_experts_、 num_experts_
    assert(redundant_expert_mapping_shape_[0] == num_layers_);
    assert(redundant_expert_mapping_shape_[1] == max_redundant_count_);
    assert(redundant_expert_mapping_shape_[2] == num_experts_);

    int redundant_count = redundant_expert_mapping_shape_[1];

    // 初始化一维映射数组, 初始值为 -1，表示未分配位置
    size_t total_size = num_layers_ * max_redundant_count_ * num_experts_;
    std::vector<int32_t> temp_mapping(total_size, -1);

    // 为每个层、每次冗余分配专家 ID 到位置的映射
    for (size_t layer = 0; layer < num_layers_; layer++) {
        for (int cnt = 0; cnt < redundant_count; cnt++) {
            // 为当前层的当前冗余实例构建独立的映射表
            for (int expert = 0; expert < num_experts_; expert++) {
                // 专家 epid 在该冗余实例中分配到位置 position, 这里用简单的round robin的方式
                int32_t redundancy_idx = cnt % get_redundant_count(layer, expert);
                size_t index = layer * redundant_count * num_experts_ + cnt * num_experts_ + expert;
                temp_mapping[index] = get_global_expert_position_id(layer, expert, redundancy_idx);
            }
        }
    }

    size_t mapping_size = num_layers_ * max_redundant_count_ * num_experts_ * sizeof(int32_t);;
    aclError ret = (*get_memcpy_fun())(
        redundant_expert_mapping_,              // HBM destination
        mapping_size,
        temp_mapping.data(),                    // Host source
        mapping_size,
        ACL_MEMCPY_HOST_TO_DEVICE
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in set_redundant_expert_mapping, error code: " +
                                 std::to_string(ret));
    }
}

// 打印映射表
void PlacementMapping::print_mapping() const {
    for (int layer = 0; layer < num_layers_; ++layer) {
        std::cout << "Layer " << layer << ":" << std::endl;
        for (int redundancy = 0; redundancy < max_redundant_count_; ++redundancy) {
            std::cout << "  Redundancy " << redundancy << " (epid -> position):" << std::endl;
            for (int epid = 0; epid < num_experts_; ++epid) {
                std::cout << "    Expert " << epid << " -> Position "
                            << get_redundant_expert_position_id(layer, redundancy, epid) << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

void PlacementMapping::construct_epid_mapping_to_position() {
    std::vector<std::vector<int32_t>> temp_redundant_count(num_layers_, std::vector<int32_t>(num_experts_, 0));
    std::vector<std::vector<std::vector<int32_t>>> temp_mapping(num_layers_, std::vector<std::vector<int32_t>>(num_experts_));

    // 输入shape [num_layers_][num_experts_][max_redundant_count_]
    max_redundant_count_ = global_expert_mapping_shape_[2];

    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        int32_t position_id = 0;
        for (int32_t rank = 0; rank < world_size_; ++rank) {
            for (int32_t expert = 0; expert < num_experts_; ++expert) {
                if (placement_pattern_vector_[rank][layer][expert] == 1) {
                    int redundant_count_this_expert = get_redundant_count(layer, expert);
                    assert(redundant_count_this_expert < max_redundant_count_);
                    set_global_expert_position_id(layer, expert, redundant_count_this_expert, position_id);

                    redundant_count_this_expert++;
                    set_redundant_count(layer, expert, redundant_count_this_expert);
                    position_id++;
                }
            }
        }
    }
}

int32_t PlacementMapping::get_redundant_expert_position_id(int32_t layer_id, int32_t redundancy, int32_t expert_id) const{
    // 边界检查
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " + std::to_string(layer_id));
    }
    if (redundancy < 0 || redundancy >= max_redundant_count_) {
        throw std::out_of_range("redundancy out of range: " + std::to_string(redundancy));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " + std::to_string(expert_id));
    }

    // 计算三维数组的线性化偏移量
    int64_t offset = (layer_id * max_redundant_count_ * num_experts_ ) +
                     (redundancy * num_experts_) + expert_id;

    int32_t value;
    aclError ret = (*get_memcpy_fun())(
        &value,                                  // Host destination
        sizeof(int32_t),
        redundant_expert_mapping_ + offset,      // HBM source
        sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in get_redundant_expert_position_id at layer " +
                                 std::to_string(layer_id) + ", expert " +
                                 std::to_string(expert_id) + ", redundancy " +
                                 std::to_string(redundancy) + ", error code: " +
                                 std::to_string(ret));
    }
    return value;
}

int32_t PlacementMapping::get_global_expert_position_id(int32_t layer_id, int32_t expert_id, int32_t index) {
    // 边界检查
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " + std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " + std::to_string(expert_id));
    }
    if (index < 0 || index >= max_redundant_count_) {
        throw std::out_of_range("index out of range: " + std::to_string(index));
    }

    // 计算三维数组的线性化偏移量
    int64_t offset = (layer_id * num_experts_ * max_redundant_count_) +
                     (expert_id * max_redundant_count_) + index;

    int32_t value;
    aclError ret = (*get_memcpy_fun())(
        &value,                                  // Host destination
        sizeof(int32_t),
        global_expert_mapping_ + offset,         // HBM source
        sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in get_global_expert_mapping at layer " +
                                 std::to_string(layer_id) + ", expert " +
                                 std::to_string(expert_id) + ", index " +
                                 std::to_string(index) + ", error code: " +
                                 std::to_string(ret));
    }
    return value;
}

void PlacementMapping::set_global_expert_position_id(
    int32_t layer_id,
    int32_t expert_id,
    int32_t index,
    int32_t new_value
) {
    // 边界检查
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " + std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " + std::to_string(expert_id));
    }
    if (index < 0 || index >= max_redundant_count_) {
        throw std::out_of_range("index out of range: " + std::to_string(index));
    }

    // 计算偏移量
    int64_t offset = (layer_id * num_experts_ * max_redundant_count_) +
                     (expert_id * max_redundant_count_) + index;

    aclError ret = (*get_memcpy_fun())(
        global_expert_mapping_ + offset,         // HBM destination
        sizeof(int32_t),
        &new_value,                             // Host source
        sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in set_global_expert_mapping at layer " +
                                 std::to_string(layer_id) + ", expert " +
                                 std::to_string(expert_id) + ", index " +
                                 std::to_string(index) + ", error code: " +
                                 std::to_string(ret));
    }
}

// 读取 redundant_count_per_expert_[layer][expert] 的值
int32_t PlacementMapping::get_redundant_count(int32_t layer_id, int32_t expert_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " + std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " + std::to_string(expert_id));
    }

    int32_t offset = layer_id * num_experts_ + expert_id;
    int32_t value;
    aclError ret = (*get_memcpy_fun())(
        &value,                                  // Host destination
        sizeof(int32_t),
        redundant_count_per_expert_ + offset,    // NPU source
        sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in get_redundant_count at layer " +
                                 std::to_string(layer_id) + ", expert " +
                                 std::to_string(expert_id) + ", error code: " +
                                 std::to_string(ret));
    }
    return value;
}

// 设置 redundant_count_per_expert_[layer][expert] 的值
void PlacementMapping::set_redundant_count(int32_t layer_id, int32_t expert_id, int32_t new_value) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("layer_id out of range: " + std::to_string(layer_id));
    }
    if (expert_id < 0 || expert_id >= num_experts_) {
        throw std::out_of_range("expert_id out of range: " + std::to_string(expert_id));
    }

    int32_t offset = layer_id * num_experts_ + expert_id;
    aclError ret = (*get_memcpy_fun())(
        redundant_count_per_expert_ + offset,    // NPU destination
        sizeof(int32_t),
        &new_value,                             // Host source
        sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE
    );
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed in set_redundant_count at layer " +
                                 std::to_string(layer_id) + ", expert " +
                                 std::to_string(expert_id) + ", error code: " +
                                 std::to_string(ret));
    }
}


// ====== RedundantExpertMapping 实现 ======

RedundantExpertMapping::RedundantExpertMapping(PlacementMapping& mapping)
    : mapping_(mapping)
{
    // 初始化冗余专家映射
    std::tie(redundant_epid_this_host_, redundant_positionid_this_host_) = construct_redundant_list_in_this_host();
    expert_mapping_freeze_ = construct_freeze_ep_mapping_to_ep_mapping();
}

RedundantExpertMapping::~RedundantExpertMapping()
{
    // 析构函数留空
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
RedundantExpertMapping::construct_redundant_list_in_this_host() {
    // Calculate the device range for the current host
    int host_id = mapping_.rank_ / mapping_.num_devices_per_host_;
    int device_id_start = host_id * mapping_.num_devices_per_host_;
    int device_id_end = std::min(mapping_.world_size_, (host_id + 1) * mapping_.num_devices_per_host_);

    num_redundant_eps_per_host_ = (device_id_end - device_id_start) *
                                 (mapping_.num_deploy_experts_per_device_ - mapping_.num_experts_per_device_);
    assert(num_redundant_eps_per_host_ >= 0);

    // Initialize redundant expert arrays
    std::vector<std::vector<int>> redundant_epid_this_host(mapping_.num_layers_,
                                  std::vector<int>(num_redundant_eps_per_host_, -1));
    std::vector<std::vector<int>> redundant_positionid_this_host(mapping_.num_layers_,
                                  std::vector<int>(num_redundant_eps_per_host_, -1));

    assert(host_id >= 0);
    assert(device_id_start >= 0);
    assert(device_id_end >= device_id_start + 1);

    // Identify redundant experts for each layer
    for (int layerid = 0; layerid < mapping_.num_layers_; layerid++) {
        int count = 0;
        for (int deviceid = device_id_start; deviceid < device_id_end; deviceid++) {
            for (int epid = 0; epid < mapping_.num_experts_; epid++) {
                // An expert is redundant if it's placed on a device but doesn't "belong" to that device
                if (mapping_.placement_pattern_vector_[deviceid][layerid][epid] == 1 &&
                    ((epid / mapping_.num_experts_per_device_) != deviceid)) {

                    redundant_epid_this_host[layerid][count] = epid;

                    // Calculate sum of placement pattern up to epid
                    int sum = 0;
                    for (int i = 0; i < epid; i++) {
                        sum += mapping_.placement_pattern_vector_[deviceid][layerid][i];
                    }

                    redundant_positionid_this_host[layerid][count] =
                        mapping_.num_deploy_experts_per_device_ * deviceid + sum;

                    count++;
                    assert(count <= num_redundant_eps_per_host_);
                }
            }
        }
    }

    return std::make_pair(redundant_epid_this_host, redundant_positionid_this_host);
}

std::vector<std::vector<int>> RedundantExpertMapping::construct_freeze_ep_mapping_to_ep_mapping() {
    // Create a zero-initialized placement pattern with same dimensions
    std::vector<std::vector<std::vector<int>>> placement_pattern_freeze(
        mapping_.world_size_,
        std::vector<std::vector<int>>(
            mapping_.num_layers_,
            std::vector<int>(mapping_.num_experts_, 0)
        )
    );

    // Iterate through all layers
    for (int layerid = 0; layerid < mapping_.num_layers_; layerid++) {
        // Find the device IDs for each expert
        std::vector<std::vector<int>> expert_locations(mapping_.num_experts_);

        for (int epid = 0; epid < mapping_.num_experts_; epid++) {
            for (int deviceid = 0; deviceid < mapping_.world_size_; deviceid++) {
                if (mapping_.placement_pattern_vector_[deviceid][layerid][epid] == 1) {
                    expert_locations[epid].push_back(deviceid);
                }
            }
        }
        int count = 0;
        // Apply FREEZE mapping strategy
        for (int epid = 0; epid < mapping_.num_experts_; epid++) {
            const std::vector<int>& devices = expert_locations[epid];
            if (devices.size() == 1) {
                // If expert is only on one device, keep it there
                placement_pattern_freeze[devices[0]][layerid][epid] = 1;
                count++;
                assert(count <= mapping_.num_experts_);  // The Frozen part each layer should not exceed the  num_experts_.
            } else if (devices.size() >= 2) {
                for (int device_id : devices) {
                    // Apply FREEZE mapping - place expert on its "home" device
                    if (epid / (mapping_.num_experts_ / mapping_.world_size_) == device_id) {
                        placement_pattern_freeze[device_id][layerid][epid] = 1;
                        count++;
                    }
                }
            }
        }
        assert(count == mapping_.num_experts_); // The Frozen part should contain all the num_experts_.
    }

    // Convert placement pattern to expert mapping
    return mapping_.construct_placement_pattern_to_ep_mapping(placement_pattern_freeze);
}

void RedundantExpertMapping::update_Redundant_Expert_Mapping(int layer_id, int target_position, int source_epid) {
    // 检查输入参数是否有效
    if (layer_id < 0 || layer_id >= mapping_.num_layers_) {
        throw std::out_of_range("layer_id out of range");
    }
    if (target_position < 0 || target_position >= mapping_.num_deploy_experts_) {
        throw std::out_of_range("target_position out of range");
    }
    if (source_epid < 0 || source_epid >= mapping_.num_experts_) {
        throw std::out_of_range("source_epid out of range");
    }

    // 获取当前层的数据引用，避免多次索引，提高可读性
    auto& positions = redundant_positionid_this_host_[layer_id];
    auto& epids = redundant_epid_this_host_[layer_id];

    // 使用 std::find 查找目标位置
    auto it = std::find(positions.begin(), positions.end(), target_position);

    if (it != positions.end()) {
        // 计算索引并更新对应的 epid
        epids[std::distance(positions.begin(), it)] = source_epid;
    } else {
        throw std::out_of_range("GG, the provided position is not correct..");
    }
}