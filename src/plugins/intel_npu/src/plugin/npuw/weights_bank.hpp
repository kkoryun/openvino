// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "lazy_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "orc.hpp"
#include "orc/schema_npuw.hpp"

namespace ov {
namespace npuw {
// Forward declaration
class LLMCompiledModel;
class CompiledModel;
namespace weights {

class Bank {
public:
    static constexpr ov::npuw::orc::TypeId kOrcType =
        static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::WeightsBank::ID);
    // Version 0 is the frozen baseline on the wire. Any further layout changes
    // must be introduced through a new versioned payload rather than by mutating v0.
    static constexpr ov::npuw::orc::Version kOrcVersion = 0u;

    Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name);

    // Register LazyTensor in a bank if it's not there. Returns LazyTensor's unique id.
    // subgraph_id uniquely identifies the submodel/subgraph across all CompiledModel instances that may
    // share this Bank (e.g. prefill/kvcache/lm_head for LLM pipelines) - used to track which weights are
    // shared/reused across subgraphs. Recommended format: "<compiled_model_name>#<submodel_idx>".
    int64_t registerLT(const LazyTensor& tensor, const std::string& device, const std::string& subgraph_id);

    // Get registered, allocated and evaluated tensor on a specified device
    ov::Tensor get(int64_t uid, const std::string& device);

    // Evaluate and allocate all LazyTensors in the bank
    void evaluate_and_allocate();

    bool is_remote(int64_t uid) const;

    std::string get_name() const;

    // Logs, for every tensor uid registered by more than one subgraph, the full list
    // of subgraph indices that share it. Helps understand cross-subgraph weight reuse.
    void log_weight_sharing_summary() const;

private:
    friend class ov::npuw::LLMCompiledModel;
    friend class ov::npuw::CompiledModel;
    friend void ov::npuw::orc::serialize(ov::npuw::orc::Stream& stream, ov::npuw::weights::Bank& var);

    struct StoredTensor {
        LazyTensor lt;
        ov::Tensor tensor;
    };
    // Bank for specified device and their allocated memory
    struct DeviceBank {
        std::unordered_map<int64_t, StoredTensor> storage;
        std::unordered_map<LazyTensor, int64_t, LazyTensor::Hash> registered_tensors;
    };
    std::unordered_map<std::string, DeviceBank> m_device_banks;

    // Tracks, for each registered tensor uid, the list of subgraph ids that registered it.
    // A uid with more than one entry means that weight is physically shared/reused across subgraphs.
    std::unordered_map<int64_t, std::vector<std::string>> m_uid_subgraphs;

    void evaluate_cpu(DeviceBank& device_bank, const std::vector<LazyTensor>& to_process);
    void evaluate_and_allocate_on_device(DeviceBank& device_bank,
                                         const std::vector<LazyTensor>& to_process,
                                         const std::string& device);

    void serialize(ov::npuw::orc::Stream& stream);
    void read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device);

    mutable std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::string m_alloc_device;
    int64_t uid_count = 0;
    std::string m_bank_name;
};

std::shared_ptr<Bank> bank(const std::string& bank_name,
                           const std::shared_ptr<const ov::ICore>& core,
                           const std::string& alloc_device);

}  // namespace weights
}  // namespace npuw
}  // namespace ov
