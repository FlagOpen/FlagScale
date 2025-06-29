// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef PLACEMENT_MAPPING_H
#define PLACEMENT_MAPPING_H
#include <acl/acl.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory> // 用于 std::unique_ptr

const int dimensions_of_expert_mapping_layers_experts = 2;  // expert_mapping的维度数量，默认是2： num_layers * num_experts
const int dimensions_of_placement_pattern_devices_layers_experts = 3;  //  Placement_pattern的维度数量，默认是3： num_devices * num_layers * num_experts

const int NUM_DIM_OF_MAPPING = 3; // global_expert_mapping的维度数量，默认是3： [num_layers_][num_experts_][max_redundant_count_]
const int NUM_DIM_OF_COUNT = 2; // redundant_count_per_expert的维度数量，默认是3： [num_layers_][num_experts_]

// 前向声明
class RedundantExpertMapping;

/**
 * @class PlacementMapping
 * @brief Manages the mapping and placement of expert models across distributed computing devices
 *
 * This class handles the distribution and redundancy of expert models in a multi-device
 * environment for large language model inference, with support for redundancy and remapping.
 */
class PlacementMapping
{
protected:
    int rank_;                        // Global device ID in the distributed system
    int world_size_;                  // Total number of devices in the distributed system
    int num_experts_;                 // Number of logical experts (e.g., 256)
    int num_deploy_experts_;          // Number of physical deployed experts including redundant ones (e.g., 256 + 16)
    int num_deploy_experts_per_rank_; // Number of physical deployed experts per rank, including redundant ones (e.g., 17 + 1)

    int num_devices_per_host_;        // Number of devices on each host/node
    int num_layers_;                  // Number of model layers that have experts
    int num_experts_per_device_;      // Number of experts assigned to each device (without redundancy)
    int num_deploy_experts_per_device_; // Number of experts assigned to each device (with redundancy)

    int32_t * expert_mapping_;        // HBM pointer to expert mapping data (maps expert ID to position)
    int64_t expert_mapping_shape_[2]; // Shape of expert mapping tensor [num_layers, num_experts]
    int expert_mapping_dtype_;        // Data type of expert mapping (e.g., 3 for int32_t, 4 for int64_t)

    int32_t * placement_pattern_;     // DRAM pointer to placement pattern data
    int64_t placement_pattern_shape_[3]; // Shape of placement pattern tensor [num_layers, num_deploy_experts, ?]
    int placement_pattern_dtype_;     // Data type of placement pattern tensor

    int32_t * redundant_expert_mapping_;     // HBM pointer to expert mapping data (maps expert ID to position)
    int64_t redundant_expert_mapping_shape_[3]; // Shape of expert mapping tensor [num_layers, max_redundant_count_, num_experts]

    int32_t * global_expert_mapping_;        // HBM pointer to expert mapping data (maps expert ID to position)
    int64_t global_expert_mapping_shape_[3]; // Shape of expert mapping tensor [num_layers, num_experts, max_redundant_count_]

    int32_t * redundant_count_per_expert_;        // HBM pointer to expert redundant count data (redundant num each expert)
    int64_t redundant_count_per_expert_shape_[2]; // Shape of expert mapping tensor [num_layers, num_experts]
    int32_t max_redundant_count_ ;

    std::vector<std::vector<std::vector<int>>> placement_pattern_vector_; // C++ representation of placement pattern

    // Mappings for expert placement management
    std::vector<std::vector<int>> position_to_epid_;           // Reverse mapping from position to expert ID

    // 内部组件：冗余专家映射
    std::unique_ptr<RedundantExpertMapping> redundant_mapping_; // 可选的冗余专家映射组件


    /**
     * @struct PlacementData
     * @brief Structure to hold placement data read from binary file
     */
    struct PlacementData {
        int32_t shape[3]; // num_devices * num_layers * num_experts
        int32_t data[];   // Flexible array member
    };



private:
    /**
     * @brief Converts tensor data to 3D vector of integers
     * @return 3D vector containing placement pattern data
     */
    std::vector<std::vector<std::vector<int>>> torch_tensor_to_3d_vector_int32();


    void construct_epid_mapping_to_position();
    void construct_epid_mapping_to_position_using_max_glabal_offset();
    void construct_per_redundancy_epid_mapping_to_position();

    /**
     * @brief Builds a mapping from position IDs to expert IDs
     * @return 2D vector mapping positions to expert IDs for each layer
     */
    std::vector<std::vector<int>> construct_position_mapping_to_epid();

    /**
     * @brief Converts placement pattern to expert mapping
     * @param placement_pattern_here The placement pattern to convert
     * @return 2D vector with expert-to-position mapping derived from the pattern
     */
    std::vector<std::vector<int>> construct_placement_pattern_to_ep_mapping(const std::vector<std::vector<std::vector<int>>>& placement_pattern_here);

    /**
     * @brief Load placement pattern from binary file
     * @param filename Path to the binary file
     * @return 3D vector containing the loaded placement pattern
     */
    std::vector<std::vector<std::vector<int>>> load_placement_pattern_from_file(const char* filename);



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// public:
//     /**
//      * @brief Constructor for PlacementMapping
//      * @param rank Current device's rank in the distributed system
//      * @param num_devices_per_host Number of devices per host/node
//      * @param expert_mapping_ptr Pointer to expert mapping data in memory
//      * @param shape Shape of the expert mapping tensor
//      * @param dtype Data type of the expert mapping tensor
//      * @param placement_pattern_ptr Pointer to placement pattern data in memory
//      * @param placement_shape Shape of the placement pattern tensor
//      * @param placement_dtype Data type of the placement pattern tensor
//      * @param enable_redundant_mapping Whether to enable redundant expert mapping functionality
//      */
//     PlacementMapping(int rank, int num_devices_per_host,
//         int32_t * expert_mapping_ptr, int64_t shape[2], int dtype,
//         int32_t * placement_pattern_ptr, int64_t placement_shape[3], int placement_dtype,
//         bool enable_redundant_mapping = true);
public:
    /**
     * @brief Constructor for PlacementMapping
     * @param rank Current device's rank in the distributed system
     * @param num_devices_per_host Number of devices per host/node
     * @param expert_mapping_ptr Pointer to expert mapping data in memory
     * @param shape Shape of the expert mapping tensor
     * @param dtype Data type of the expert mapping tensor
     * @param placement_pattern_ptr Pointer to placement pattern data in memory (can be nullptr if using file)
     * @param placement_shape Shape of the placement pattern tensor (used if placement_pattern_ptr is not nullptr)
     * @param placement_dtype Data type of the placement pattern tensor (used if placement_pattern_ptr is not nullptr)
     * @param placement_pattern_filename Path to binary file containing placement pattern (used if placement_pattern_ptr is nullptr)
     * @param enable_redundant Whether to enable redundant expert mapping functionality
     */
    PlacementMapping(int rank, int num_devices_per_host,
        int32_t * expert_mapping_ptr, int64_t shape[dimensions_of_expert_mapping_layers_experts], int dtype,
        int32_t * placement_pattern_ptr, int64_t placement_shape[dimensions_of_placement_pattern_devices_layers_experts], int placement_dtype,
        const char* placement_pattern_filename = "",
        bool enable_redundant = true);

    PlacementMapping(const std::string& placement_pattern_filename, int rank, int num_devices_per_host,
        size_t redundant_expert_mapping_ptr, std::vector<int64_t> redundant_mapping_shape,
        size_t global_expert_mapping_ptr, std::vector<int64_t> global_mapping_shape,
        size_t expert_redundant_count_ptr, std::vector<int64_t> count_shape,
        size_t placement_pattern_ptr, std::vector<int64_t> pattern_shape);

    //for UT test
    PlacementMapping(std::vector<std::vector<std::vector<int>>> &placement_pattern_vector, int rank, int num_devices_per_host,
        int32_t * redundant_expert_mapping_ptr, int64_t redundant_mapping_shape[NUM_DIM_OF_MAPPING],
        int32_t * global_expert_mapping_ptr, int64_t global_mapping_shape[NUM_DIM_OF_MAPPING],
        int32_t * expert_redundant_count_ptr, int64_t count_shape[NUM_DIM_OF_COUNT]);

    /**
     * @brief Destructor for PlacementMapping
     */
    virtual ~PlacementMapping();



    /**
     * @brief Reassigns an expert to a new position
     * @param layer_id The layer ID where the expert exists
     * @param expert_id The ID of the expert to move
     * @param new_position The new position to assign to the expert
     */
    void change_pos_id(int layer_id, int expert_id, int new_position);

    /**
     * @brief Retrieves the position ID for a given expert (primarily for unit testing)
     * @param layer_id The layer ID to check
     * @param expert_id The expert ID to look up
     * @return The position ID where the expert is located
     */
    int read_pos_id(int layer_id, int expert_id);

    // Getter methods for class properties
    int get_rank() { return rank_; }
    int get_world_size() { return world_size_; }
    int get_num_layers() { return num_layers_; }
    int get_num_experts() { return num_experts_; }
    int get_num_deploy_experts() { return num_deploy_experts_; }
    int get_num_devices_per_host() { return num_devices_per_host_; }
    int32_t * get_redundant_expert_mapping() { return redundant_expert_mapping_; }
    void print_mapping() const;
    /**
     * @brief Gets the default mapping position for a specific expert
     * @param layer_idx The layer index
     * @param expert_id The expert ID
     * @return The position ID from the frozen mapping
     */
    int get_default_mapping_position(int layer_idx, int expert_id);

    // Getters for mapping data structures
    std::vector<std::vector<int>> get_position_to_epid() { return position_to_epid_; }



    /**
     * @brief Updates the position-to-expert mapping
     * @param layer_id The layer ID to update
     * @param target_position The position ID to update
     * @param source_epid The expert ID to associate with the position
     */
    void update_Position_To_Expert_Mapping(int layer_id, int target_position, int source_epid);

    /**
     * @brief Check if redundant expert mapping is enabled
     * @return true if redundant expert mapping is enabled, false otherwise
     */
    bool is_redundant_mapping_enabled() const { return redundant_mapping_ != nullptr; }

    /**
     * @brief Enable redundant expert mapping functionality
     * @return true if successfully enabled, false otherwise
     */
    bool enable_redundant_mapping();

    /**
     * @brief Disable redundant expert mapping functionality
     */
    void disable_redundant_mapping();

    /**
     * @brief Gets the list of redundant expert IDs on this host
     * @return 2D vector of redundant expert IDs, empty if not enabled
     */
    std::vector<std::vector<int>> get_redundant_epid_this_host();

    /**
     * @brief Gets the list of redundant positions on this host
     * @return 2D vector of redundant position IDs, empty if not enabled
     */
    std::vector<std::vector<int>> get_redundant_positionid_this_host();


    /**
     * @brief Updates the mapping for redundant experts
     * @param layer_id The layer ID to update
     * @param target_position The target position ID
     * @param source_epid The source expert ID to place at the target position
     * @throws std::runtime_error if redundant expert mapping is not enabled
     */
    void update_Redundant_Expert_Mapping(int layer_id, int target_position, int source_epid);

    int32_t get_global_expert_position_id(int32_t layer_id, int32_t expert_id, int32_t index);
    void set_global_expert_position_id(int32_t layer_id, int32_t expert_id, int32_t index, int32_t new_value);
    int32_t get_redundant_count(int32_t layer_id, int32_t expert_id);
    void set_redundant_count(int32_t layer_id, int32_t expert_id, int32_t new_value);
    int32_t get_redundant_expert_position_id(int32_t layer_id, int32_t redundancy, int32_t expert_id) const;

    // 允许子类访问内部数据的友元声明
    friend class RedundantExpertMapping;
};

/**
 * @class RedundantExpertMapping
 * @brief Internal component that manages redundant expert mappings
 *
 * This class is used internally by PlacementMapping to handle redundant expert mappings.
 */
class RedundantExpertMapping {
private:
    PlacementMapping& mapping_;  // 引用到PlacementMapping对象

    int num_redundant_eps_per_host_;  // Number of redundant expert placements per host
    std::vector<std::vector<int>> redundant_epid_this_host_;   // IDs of redundant experts on current host
    std::vector<std::vector<int>> redundant_positionid_this_host_; // Position IDs of redundant experts on current host
    std::vector<std::vector<int>> expert_mapping_freeze_;      // Fixed expert-to-position mapping (immutable reference)

public:
    /**
     * @brief Constructor for RedundantExpertMapping
     * @param mapping Reference to the parent PlacementMapping object
     */
    RedundantExpertMapping(PlacementMapping& mapping);

    /**
     * @brief Destructor for RedundantExpertMapping
     */
    ~RedundantExpertMapping();

    /**
     * @brief Gets the list of redundant expert IDs on this host
     * @return 2D vector of redundant expert IDs
     */
    std::vector<std::vector<int>> get_redundant_epid_this_host() const { return redundant_epid_this_host_; }

    /**
     * @brief Gets the list of redundant positions on this host
     * @return 2D vector of redundant position IDs
     */
    std::vector<std::vector<int>> get_redundant_positionid_this_host() const { return redundant_positionid_this_host_; }

    /**
     * @brief Gets the default mapping position for a specific expert
     * @param layer_idx The layer index
     * @param expert_id The expert ID
     * @return The position ID from the frozen mapping
     */
    int get_default_mapping_position(int layer_idx, int expert_id) const { return expert_mapping_freeze_[layer_idx][expert_id]; }

    /**
     * @brief Creates lists of redundant experts and their positions on the current host
     * @return A pair of vectors: (redundant expert IDs, redundant position IDs)
     */
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> construct_redundant_list_in_this_host();

    /**
     * @brief Creates a frozen copy of the expert mapping
     * @return 2D vector containing the frozen expert-to-position mapping
     */
    std::vector<std::vector<int>> construct_freeze_ep_mapping_to_ep_mapping();

    /**
     * @brief Updates the mapping for redundant experts
     * @param layer_id The layer ID to update
     * @param target_position The target position ID
     * @param source_epid The source expert ID to place at the target position
     */
    void update_Redundant_Expert_Mapping(int layer_id, int target_position, int source_epid);
};

#endif // PLACEMENT_MAPPING_H