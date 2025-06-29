// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef PLACEMENT_OPTIMIZER_H
#define PLACEMENT_OPTIMIZER_H

#include <vector>          
#include <string>           
#include <tuple>            
#include <unordered_set>    
#include <algorithm>        
#include <stdexcept>
#include "expert_activation.h"  
#include "placement_mapping.h"  

/**
 * @brief Class for optimizing expert placement across devices and hosts.
 * 
 * Manages the placement of experts in a distributed system, optimizing their distribution
 * based on activation frequencies and redundancy.
 */
class PlacementOptimizer {
private:
    PlacementMapping* placement_mapping_;         ///< Pointer to placement mapping data
    ClusterActivation* clusterActivation_;        ///< Pointer to cluster activation data
    int num_layers_;                              ///< Number of layers in the model
    int rank_;                                    ///< Rank of the current process
    int world_size_;                              ///< Total number of processes (world size)
    int num_deploy_experts_;                      ///< Total number of experts to deploy
    int num_experts_;                             ///< Total number of experts in the model
    int num_devices_per_host_;                    ///< Number of devices per host
    int num_deploy_experts_per_device_;           ///< Number of experts deployed per device

    /**
     * @brief Computes host and device parameters such as positions and loads.
     * 
     * @param[out] current_host Host ID for the current rank.
     * @param[out] host_start_position Starting position index for this host.
     * @param[out] host_end_position Ending position index for this host.
     * @param[out] positions_per_host Number of positions per host.
     * @param[out] host_positions Vector of position indices for this host.
     * @param[out] device_loads Vector of load values for each device.
     * @throws std::invalid_argument If device configuration is invalid.
     */
    void compute_host_device_params(
        int& current_host,
        int& host_start_position,
        int& host_end_position,
        int& positions_per_host,
        std::vector<int>& host_positions,
        std::vector<int64_t>& device_loads
    );

    /**
     * @brief Computes the load on each device based on layer frequency.
     * 
     * @param[in] layer_freq Vector of activation counts for each position.
     * @param[in] host_start_position Starting position index for this host.
     * @param[out] device_loads Vector to store computed loads.
     * @throws std::invalid_argument If input parameters are invalid.
     */
    void compute_device_loads(
        const std::vector<int64_t>& layer_freq,
        int host_start_position,
        std::vector<int64_t>& device_loads
    );

    /**
     * @brief Retrieves redundant positions within the current host for a given layer.
     * 
     * @param[in] layer_idx_moe Layer index for Mixture of Experts (MoE).
     * @return std::vector<int> Vector of redundant position IDs.
     * @throws std::out_of_range If layer_idx_moe is invalid.
     */
    std::vector<int> get_host_redundant_positions(
        int layer_idx_moe
    );

    /**
     * @brief Sorts redundant positions by their loads in ascending order.
     * 
     * @param[in] host_redundant_positions Vector of redundant position IDs.
     * @param[in] layer_freq Vector of activation counts for each position.
     * @return std::vector<std::pair<int, int>> Vector of (position, load) pairs sorted by load.
     */
    std::vector<std::pair<int, int>> sort_redundant_loads(
        const std::vector<int>& host_redundant_positions,
        const std::vector<int64_t>& layer_freq
    );

    /**
     * @brief Finds a replacement for a redundant expert.
     * 
     * @param[in] redundant_loads Sorted vector of (position, load) pairs.
     * @param[in] host_redundant_positions Vector of redundant position IDs.
     * @param[in] layer_idx_moe Layer index for MoE.
     * @param[in] layer_freq Vector of activation counts.
     * @param[in] host_start_position Starting position index for this host.
     * @param[in] valid_ep_start Start of valid expert ID range.
     * @param[in] valid_ep_end End of valid expert ID range.
     * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement found.
     * @throws std::out_of_range If layer_idx_moe is invalid.
     */
    std::tuple<int, int, int> find_replacement(
        const std::vector<std::pair<int, int>>& redundant_loads,
        const std::vector<int>& host_redundant_positions,
        int layer_idx_moe,
        const std::vector<int64_t>& layer_freq,
        int host_start_position
    );

    /**
     * @brief Replaces a redundant expert in the given layer.
     * 
     * @param[in] layer_idx_moe Layer index for MoE.
     * @param[in] layer_freq Vector of activation counts.
     * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement possible.
     */
    std::tuple<int, int, int> replace_redundant_ep(
        int layer_idx_moe,
        const std::vector<int64_t>& layer_freq
    );

public:
    /**
     * @brief Constructor for PlacementOptimizer.
     * 
     * Initializes the optimizer with placement and activation data.
     * 
     * @param[in] placement_mapping Pointer to PlacementMapping object.
     * @param[in] clusterActivation Pointer to ClusterActivation object.
     * @throws std::runtime_error If either pointer is null.
     */
    PlacementOptimizer(PlacementMapping* placement_mapping, ClusterActivation* clusterActivation);

    /**
     * @brief Default destructor.
     * 
     * No special cleanup is needed as the class does not own the pointers.
     */
    ~PlacementOptimizer() = default;

    /**
     * @brief Retrieves the frequency status for a specific layer.
     * 
     * @param[in] layer Layer index to query.
     * @return std::vector<int64_t> Vector of activation counts for each position in the layer.
     * @throws std::out_of_range If layer index is invalid.
     */
    std::vector<int64_t> get_layer_freq_status(int layer);

    /**
     * @brief Optimizes expert placement for the specified layer.
     * 
     * @param[in] layer_id Layer index to optimize.
     * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no optimization needed.
     * @throws std::out_of_range If layer_id is invalid.
     */
    std::tuple<int, int, int> optimize(int layer_id);

    /** @brief Gets the number of layers in the model. @return int Number of layers. */
    int get_num_layers() const { return num_layers_; }
    /** @brief Gets the rank of the current process. @return int Rank. */
    int get_rank() const { return rank_; }
    /** @brief Gets the total number of processes. @return int World size. */
    int get_world_size() const { return world_size_; }
    /** @brief Gets the total number of experts to deploy. @return int Number of deployed experts. */
    int get_num_deploy_experts() const { return num_deploy_experts_; }
    /** @brief Gets the total number of experts in the model. @return int Number of experts. */
    int get_num_experts() const { return num_experts_; }
    /** @brief Gets the number of devices per host. @return int Number of devices per host. */
    int get_num_devices_per_host() const { return num_devices_per_host_; }
    /** @brief Gets the number of experts deployed per device. @return int Number of experts per device. */
    int get_num_deploy_experts_per_device() const { return num_deploy_experts_per_device_; }

    /**
     * @brief Public wrapper for compute_host_device_params.
     * 
     * @param[out] current_host Host ID for the current rank.
     * @param[out] host_start_position Starting position index for this host.
     * @param[out] host_end_position Ending position index for this host.
     * @param[out] positions_per_host Number of positions per host.
     * @param[out] host_positions Vector of position indices for this host.
     * @param[out] device_loads Vector of load values for each device.
     */
    void get_host_device_params(
        int& current_host,
        int& host_start_position,
        int& host_end_position,
        int& positions_per_host,
        std::vector<int>& host_positions,
        std::vector<int64_t>& device_loads
    ) {
        compute_host_device_params(current_host, host_start_position, host_end_position,
                                 positions_per_host, host_positions, device_loads);
    }

    /**
     * @brief Public wrapper for compute_device_loads.
     * 
     * @param[in] layer_freq Vector of activation counts for each position.
     * @param[in] host_start_position Starting position index for this host.
     * @param[out] device_loads Vector to store computed loads.
     */
    void get_device_loads(
        const std::vector<int64_t>& layer_freq,
        int host_start_position,
        std::vector<int64_t>& device_loads
    ) {
        compute_device_loads(layer_freq, host_start_position, device_loads);
    }

    /**
     * @brief Public wrapper for get_host_redundant_positions (with unused host_positions parameter).
     * 
     * @param[in] layer_idx_moe Layer index for MoE.
     * @param[in] host_positions Vector of position indices for this host (unused).
     * @return std::vector<int> Vector of redundant position IDs.
     */
    std::vector<int> get_redundant_positions(
        int layer_idx_moe
    ) {
        return get_host_redundant_positions(layer_idx_moe);
    }

    /**
     * @brief Public wrapper for sort_redundant_loads.
     * 
     * @param[in] host_redundant_positions Vector of redundant position IDs.
     * @param[in] layer_freq Vector of activation counts for each position.
     * @return std::vector<std::pair<int, int>> Vector of (position, load) pairs sorted by load.
     */
    std::vector<std::pair<int, int>> get_sorted_redundant_loads(
        const std::vector<int>& host_redundant_positions,
        const std::vector<int64_t>& layer_freq
    ) {
        return sort_redundant_loads(host_redundant_positions, layer_freq);
    }

    /**
     * @brief Public wrapper for find_replacement.
     * 
     * @param[in] redundant_loads Sorted vector of (position, load) pairs.
     * @param[in] host_redundant_positions Vector of redundant position IDs.
     * @param[in] layer_idx_moe Layer index for MoE.
     * @param[in] layer_freq Vector of activation counts.
     * @param[in] host_start_position Starting position index for this host.
     * @param[in] valid_ep_start Start of valid expert ID range.
     * @param[in] valid_ep_end End of valid expert ID range.
     * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement found.
     */
    std::tuple<int, int, int> get_replacement(
        const std::vector<std::pair<int, int>>& redundant_loads,
        const std::vector<int>& host_redundant_positions,
        int layer_idx_moe,
        const std::vector<int64_t>& layer_freq,
        int host_start_position
    ) {
        return find_replacement(redundant_loads, host_redundant_positions, layer_idx_moe,
                              layer_freq, host_start_position);
    }

    /**
     * @brief Public wrapper for replace_redundant_ep.
     * 
     * @param[in] layer_idx_moe Layer index for MoE.
     * @param[in] layer_freq Vector of activation counts.
     * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement possible.
     */
    std::tuple<int, int, int> get_redundant_ep_replacement(
        int layer_idx_moe,
        const std::vector<int64_t>& layer_freq
    ) {
        return replace_redundant_ep(layer_idx_moe, layer_freq);
    }
};

#endif // PLACEMENT_OPTIMIZER_H