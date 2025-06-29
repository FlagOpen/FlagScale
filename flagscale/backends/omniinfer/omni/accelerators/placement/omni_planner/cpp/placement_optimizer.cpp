// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <iostream>         
#include <stdexcept>        
#include <algorithm>        
#include <unordered_set>    
#include <numeric>          
#include "expert_activation.h" 
#include "placement_optimizer.h" 

/**
 * @brief Constructor for PlacementOptimizer.
 * 
 * Initializes member variables using pointers to PlacementMapping and ClusterActivation objects.
 * Throws an exception if either pointer is null.
 * 
 * @param placement_mapping Pointer to a PlacementMapping object containing placement data.
 * @param clusterActivation Pointer to a ClusterActivation object containing activation data.
 * @throws std::runtime_error If placement_mapping or clusterActivation is null.
 */
PlacementOptimizer::PlacementOptimizer(PlacementMapping* placement_mapping, ClusterActivation* clusterActivation)
    : placement_mapping_(placement_mapping),
      clusterActivation_(clusterActivation),
      num_layers_(placement_mapping ? placement_mapping->get_num_layers() : 0),
      rank_(placement_mapping ? placement_mapping->get_rank() : 0),
      world_size_(placement_mapping ? placement_mapping->get_world_size() : 0),
      num_deploy_experts_(placement_mapping ? placement_mapping->get_num_deploy_experts() : 0),
      num_experts_(placement_mapping ? placement_mapping->get_num_experts() : 0),
      num_devices_per_host_(placement_mapping ? placement_mapping->get_num_devices_per_host() : 0),
      num_deploy_experts_per_device_(placement_mapping && placement_mapping->get_world_size() > 0
                                    ? (placement_mapping->get_num_deploy_experts() + placement_mapping->get_world_size() - 1) / placement_mapping->get_world_size()
                                    : 0)
{
    if (!placement_mapping_ || !clusterActivation_) {
        throw std::runtime_error("Invalid initialization parameters");
    }
}

/**
 * @brief Retrieves the frequency status of a specific layer across all positions.
 * 
 * Queries the activation counts for each position in the specified layer.
 * 
 * @param layer The layer index to query.
 * @return std::vector<int64_t> A vector of activation counts for each position in the layer.
 * @throws std::out_of_range If the layer index is invalid (less than 0 or greater than or equal to num_layers_).
 */
std::vector<int64_t> PlacementOptimizer::get_layer_freq_status(int layer) {
    if (layer < 0 || layer >= num_layers_) {
        throw std::out_of_range("Invalid layer: " + std::to_string(layer));
    }

    std::vector<int64_t> layer_freq(world_size_ * num_deploy_experts_per_device_, 0);

    for (int posid = 0; posid < num_deploy_experts_; ++posid) {
        layer_freq[posid] = clusterActivation_->getClusterTotalActivationCount(layer, posid);
    }
    return layer_freq;
}

/**
 * @brief Computes host and device parameters for the current rank.
 * 
 * Calculates and modifies the provided parameters based on the current rank and device configuration.
 * 
 * @param[out] current_host The host ID for the current rank.
 * @param[out] host_start_position Starting position index for this host.
 * @param[out] host_end_position Ending position index for this host.
 * @param[out] positions_per_host Number of positions per host.
 * @param[out] host_positions Vector of position indices for this host.
 * @param[out] device_loads Vector of load values for each device on this host.
 * @throws std::invalid_argument If the number of devices per host is less than or equal to 0.
 */
void PlacementOptimizer::compute_host_device_params(int& current_host, int& host_start_position,
    int& host_end_position, int& positions_per_host, std::vector<int>& host_positions,
    std::vector<int64_t>& device_loads) {
    if (num_devices_per_host_ <= 0) {
        throw std::invalid_argument("Invalid device configuration");
    }

    current_host = rank_ / num_devices_per_host_;
    positions_per_host = num_deploy_experts_per_device_ * num_devices_per_host_;
    host_start_position = num_deploy_experts_per_device_ * (current_host * num_devices_per_host_);
    host_end_position = std::min(host_start_position + positions_per_host, num_deploy_experts_);

    host_positions.resize(host_end_position - host_start_position);
    std::iota(host_positions.begin(), host_positions.end(), host_start_position);

    device_loads.assign(num_devices_per_host_, 0);
}

/**
 * @brief Computes the load on each device based on layer frequency.
 * 
 * Sums the activation counts for positions assigned to each device.
 * 
 * @param[in] layer_freq Vector of activation counts for each position.
 * @param[in] host_start_position Starting position index for this host.
 * @param[out] device_loads Vector to store computed loads for each device.
 * @throws std::invalid_argument If layer_freq is empty or device_loads size does not match num_devices_per_host_.
 */
void PlacementOptimizer::compute_device_loads(const std::vector<int64_t>& layer_freq,
    int host_start_position, std::vector<int64_t>& device_loads) {
    if (layer_freq.empty() || device_loads.size() != static_cast<size_t>(num_devices_per_host_)) {
        throw std::invalid_argument("Invalid input parameters");
    }

    for (int i = 0; i < num_devices_per_host_; ++i) {
        int start = host_start_position + i * num_deploy_experts_per_device_;
        int end = std::min(start + num_deploy_experts_per_device_, static_cast<int>(layer_freq.size()));
        device_loads[i] = std::accumulate(layer_freq.begin() + start, layer_freq.begin() + end, 0);
    }
}

/**
 * @brief Retrieves redundant position IDs for a specific layer on the current host.
 * 
 * Queries the redundant position IDs for the specified Mixture of Experts (MoE) layer.
 * 
 * @param[in] layer_idx_moe Layer index for Mixture of Experts (MoE).
 * @return std::vector<int> Vector of redundant position IDs.
 * @throws std::out_of_range If layer_idx_moe is invalid.
 */
std::vector<int> PlacementOptimizer::get_host_redundant_positions(int layer_idx_moe) {
    auto redundant_positions = placement_mapping_->get_redundant_positionid_this_host();

    if (layer_idx_moe < 0 || static_cast<size_t>(layer_idx_moe) >= redundant_positions.size()) {
        throw std::out_of_range("Invalid layer_idx_moe: " + std::to_string(layer_idx_moe));
    }
    return redundant_positions[layer_idx_moe];
}

/**
 * @brief Sorts redundant positions by their activation loads in ascending order.
 * 
 * Creates a vector of position-load pairs and sorts them by load.
 * 
 * @param[in] host_redundant_positions Vector of redundant position IDs.
 * @param[in] layer_freq Vector of activation counts for each position.
 * @return std::vector<std::pair<int, int>> Vector of pairs (position, load), sorted by load.
 */
std::vector<std::pair<int, int>> PlacementOptimizer::sort_redundant_loads(
    const std::vector<int>& host_redundant_positions, const std::vector<int64_t>& layer_freq) {
    std::vector<std::pair<int, int>> redundant_loads;
    redundant_loads.reserve(host_redundant_positions.size());

    for (int pos : host_redundant_positions) {
        if (pos >= 0 && static_cast<size_t>(pos) < layer_freq.size()) {
            redundant_loads.emplace_back(pos, layer_freq[pos]);
        }
    }

    std::sort(redundant_loads.begin(), redundant_loads.end(),
              [](auto& a, auto& b) { return a.second < b.second; });
    return redundant_loads;
}

/**
 * @brief Finds a replacement for a redundant expert.
 * 
 * Identifies a non-redundant expert to replace a redundant one, ensuring they are on different devices.
 * 
 * @param[in] redundant_loads Sorted vector of (position, load) pairs for redundant positions.
 * @param[in] host_redundant_positions Vector of redundant position IDs.
 * @param[in] layer_idx_moe Layer index for Mixture of Experts (MoE).
 * @param[in] layer_freq Vector of activation counts.
 * @param[in] host_start_position Starting position index for this host.
 * @param[in] valid_ep_start Start of valid expert ID range.
 * @param[in] valid_ep_end End of valid expert ID range.
 * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement found.
 * @throws std::out_of_range If layer_idx_moe is invalid.
 */
std::tuple<int, int, int> PlacementOptimizer::find_replacement(
    const std::vector<std::pair<int, int>>& redundant_loads,
    const std::vector<int>& host_redundant_positions, int layer_idx_moe,
    const std::vector<int64_t>& layer_freq, int host_start_position) {
    auto position_to_ep = placement_mapping_->get_position_to_epid();
    auto redundant_eps = placement_mapping_->get_redundant_epid_this_host();

    if (layer_idx_moe < 0 || static_cast<size_t>(layer_idx_moe) >= position_to_ep.size()) {
        throw std::out_of_range("Invalid layer_idx_moe");
    }
    if (redundant_loads.empty()) return {-1, -1, -1};

    std::unordered_set<int> redundant_ep_set(redundant_eps[layer_idx_moe].begin(), redundant_eps[layer_idx_moe].end());

    std::vector<std::pair<int, int>> device_loads_indexed(num_devices_per_host_);
    std::vector<int64_t> device_loads(num_devices_per_host_, 0);
    compute_device_loads(layer_freq, host_start_position, device_loads);
    for (int i = 0; i < num_devices_per_host_; ++i) {
        device_loads_indexed[i] = {i, device_loads[i]};
    }
    std::sort(device_loads_indexed.begin(), device_loads_indexed.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [dev, _] : device_loads_indexed) {
        int start = host_start_position + dev * num_deploy_experts_per_device_;
        int end = std::min(start + num_deploy_experts_per_device_, static_cast<int>(layer_freq.size()));

        std::vector<std::pair<int, int>> non_redundant_pos_loads;
        non_redundant_pos_loads.reserve(end - start);
        for (int pos = start; pos < end; ++pos) {
            int ep = position_to_ep[layer_idx_moe][pos];
            if (redundant_ep_set.find(ep) == redundant_ep_set.end()) {
                non_redundant_pos_loads.emplace_back(pos, layer_freq[pos]);
            }
        }

        std::sort(non_redundant_pos_loads.begin(), non_redundant_pos_loads.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [pos, _] : non_redundant_pos_loads) {
            int source_ep = position_to_ep[layer_idx_moe][pos];

            for (const auto& [target_pos, __] : redundant_loads) {
                int target_ep = position_to_ep[layer_idx_moe][target_pos];
                int target_device = (target_pos - host_start_position) / num_deploy_experts_per_device_;

                if (target_device == dev) continue;

                return {source_ep, target_pos, target_ep};
            }
        }
    }
    return {-1, -1, -1};
}

/**
 * @brief Replaces a redundant expert in the specified layer.
 * 
 * Attempts to replace a redundant expert based on activation counts.
 * 
 * @param[in] layer_idx_moe Layer index for Mixture of Experts (MoE).
 * @param[in] layer_freq Vector of activation counts.
 * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no replacement possible.
 */
std::tuple<int, int, int> PlacementOptimizer::replace_redundant_ep(int layer_idx_moe,
    const std::vector<int64_t>& layer_freq) {
    if (layer_freq.empty()) return {-1, -1, -1};

    int current_host, host_start, host_end, positions_per_host;
    std::vector<int> host_positions;
    std::vector<int64_t> device_loads;

    compute_host_device_params(current_host, host_start, host_end, positions_per_host, host_positions, device_loads);

    auto redundant_positions = get_host_redundant_positions(layer_idx_moe);
    if (redundant_positions.empty()) return {-1, -1, -1};

    auto redundant_loads = sort_redundant_loads(redundant_positions, layer_freq);

    return find_replacement(redundant_loads, redundant_positions, layer_idx_moe, layer_freq,
                           host_start);
}

/**
 * @brief Optimizes expert placement for the specified layer.
 * 
 * Attempts to replace a redundant expert to optimize placement for the given layer.
 * 
 * @param[in] layer_id The layer index to optimize.
 * @return std::tuple<int, int, int> Tuple (source_ep, target_pos, target_ep) or (-1, -1, -1) if no optimization needed.
 * @throws std::out_of_range If layer_id is invalid.
 */
std::tuple<int, int, int> PlacementOptimizer::optimize(int layer_id) {
    if (layer_id < 0 || layer_id >= num_layers_) {
        throw std::out_of_range("Invalid layer_id: " + std::to_string(layer_id));
    }

    return replace_redundant_ep(layer_id, get_layer_freq_status(layer_id));
}