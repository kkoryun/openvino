// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include <execinfo.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <sstream>

#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "serialization.hpp"
#include "util.hpp"

using ov::npuw::weights::Bank;

using ov::npuw::weights::LazyTensor;

namespace {
std::atomic<std::size_t> g_bank_total_bytes{0};
std::atomic<std::size_t> g_bank_total_tensors{0};
}  // namespace

class BankManager {
public:
    static BankManager& getInstance() {
        static BankManager instance;
        return instance;
    }

private:
    BankManager() {}
    BankManager(const BankManager&) = delete;
    void operator=(const BankManager&) = delete;

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name,
                                  const std::shared_ptr<const ov::ICore>& core,
                                  const std::string& alloc_device);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

Bank::Bank(const std::shared_ptr<const ov::ICore>& core, const std::string& alloc_device, const std::string& bank_name)
    : m_core(core),
      m_alloc_device(alloc_device),
      m_bank_name(bank_name) {
    LOG_ERROR("Creating weights bank: " << m_bank_name << " alloc_device=" << m_alloc_device);
    if (m_bank_name.empty()) {
        auto unique_name = ov::npuw::util::generate_random_string();
        LOG_WARN("Got an empty name for weights bank! Using a uniquely generated instead: " << unique_name);
        m_bank_name = unique_name;
    }
}

int64_t Bank::registerLT(const LazyTensor& tensor, const std::string& device, const std::string& subgraph_id) {
    auto _name = tensor.get_const_name();
    std::string _name_str = _name ? *_name : std::string("<non-const>");
    auto _offset = tensor.get_const_offset();
    std::string _offset_str = _offset ? std::to_string(*_offset) : std::string("<na>");
    auto _size = tensor.get_const_size();
    std::string _size_str = _size ? std::to_string(*_size) : std::string("<na>");
    auto _wc_info = tensor.get_const_weightless_cache_info();
    std::string _wc_str = "<na>";
    if (_wc_info) {
        std::ostringstream oss;
        oss << "original_size=" << _wc_info->original_size << " bin_offset=" << _wc_info->bin_offset
            << " original_dtype=" << _wc_info->original_dtype;
        _wc_str = oss.str();
    }

    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device_for_alloc];

    auto iter_registered = device_bank.registered_tensors.find(tensor);
    int64_t uid = -1;
    bool is_new = false;
    if (iter_registered == device_bank.registered_tensors.end()) {
        uid = uid_count++;
        device_bank.registered_tensors[tensor] = uid;
        device_bank.storage[uid] = {tensor, ov::Tensor()};
        is_new = true;
    } else {
        uid = iter_registered->second;
    }

    // Track which subgraphs share this uid, and report reuse across subgraphs.
    auto& sharers = m_uid_subgraphs[uid];
    std::string _shared_with_str = "<none>";
    if (!sharers.empty()) {
        std::ostringstream oss;
        for (std::size_t i = 0; i < sharers.size(); ++i) {
            oss << sharers[i];
            if (i + 1 < sharers.size()) {
                oss << ",";
            }
        }
        _shared_with_str = oss.str();
    }
    LOG_ERROR("Registering LazyTensor in weights bank: "
              << m_bank_name << " device=" << device << " subgraph=" << subgraph_id << " name=" << _name_str
              << " tensor=" << tensor.eval_meta().shape << " type=" << tensor.eval_meta().type
              << " offset=" << _offset_str << " size_bytes=" << _size_str << " weightless_cache[" << _wc_str << "]"
              << " uid=" << uid << " status=" << (is_new ? "NEW" : "REUSED")
              << " already_registered_by_subgraphs=[" << _shared_with_str << "]");
    sharers.push_back(subgraph_id);

    // Remember the human-readable name for this uid while we still have it - the source
    // LazyTensor's underlying Constant node may be detach()-ed shortly after this call
    // (see below), at which point the name becomes unrecoverable from the LazyTensor itself.
    if (_name && m_uid_names.find(uid) == m_uid_names.end()) {
        m_uid_names[uid] = *_name;
    }

    if (!is_new) {
        // Already registered - can safely detach the incoming tensor
        const_cast<LazyTensor&>(tensor).detach();
    }

    return uid;
}

void Bank::log_weight_sharing_summary() const {
    std::unique_lock guard(m_mutex);
    for (const auto& kv : m_uid_subgraphs) {
        const auto uid = kv.first;
        const auto& subgraphs = kv.second;
        if (subgraphs.size() <= 1) {
            continue;  // not shared across subgraphs
        }
        std::string name_str = "<unknown>";
        std::optional<std::size_t> byte_size;
        for (const auto& dev_kv : m_device_banks) {
            auto it = dev_kv.second.storage.find(uid);
            if (it != dev_kv.second.storage.end()) {
                auto n = it->second.lt.get_const_name();
                name_str = n ? *n : std::string("<non-const>");
                byte_size = it->second.lt.get_const_size();
                if (!byte_size && it->second.tensor) {
                    // Fall back to the already-materialized tensor's size (e.g. for
                    // derived/computed weights that aren't a plain Const).
                    byte_size = it->second.tensor.get_byte_size();
                }
                break;
            }
        }
        if (name_str == "<non-const>") {
            // The LazyTensor's own Constant node may have been detach()-ed by now (this
            // weight is REUSED, see registerLT()). Fall back to the name captured at
            // registration time - don't call get_name() here, it re-locks m_mutex.
            auto names_it = m_uid_names.find(uid);
            if (names_it != m_uid_names.end()) {
                name_str = names_it->second;
            }
        }
        // Skip tiny weights (e.g. norm scales/biases) to keep the summary focused on
        // the sizable, actually-interesting shared tensors. Unknown-size entries are
        // still printed since we can't tell whether they're small or not.
        constexpr std::size_t kMinBytesToLog = static_cast<std::size_t>(0.1 * 1024 * 1024);
        if (byte_size && *byte_size < kMinBytesToLog) {
            continue;
        }
        std::string size_str =
            byte_size ? std::to_string(*byte_size / float(1024 * 1024)) + " MB" : std::string("<unknown>");
        std::ostringstream oss;
        for (std::size_t i = 0; i < subgraphs.size(); ++i) {
            oss << subgraphs[i];
            if (i + 1 < subgraphs.size()) {
                oss << ",";
            }
        }
        LOG_ERROR("[WEIGHT_SHARING] bank=" << m_bank_name << " uid=" << uid << " name=" << name_str
                                            << " size=" << size_str << " subgraphs=[" << oss.str()
                                            << "] num_subgraphs=" << subgraphs.size());
    }
}

ov::Tensor Bank::get(int64_t uid, const std::string& device) {
    LOG_ERROR("Getting tensor from weights bank: " << m_bank_name << " device=" << device << " uid=" << uid);
    const std::string& device_for_alloc = m_alloc_device.empty() ? device : m_alloc_device;

    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks.at(device_for_alloc);
    auto iter_device = device_bank.storage.find(uid);

    NPUW_ASSERT(iter_device != device_bank.storage.end() && iter_device->second.tensor &&
                "Tensor should be registered and allocated first!");

    return iter_device->second.tensor;
}

struct TensorToAllocate {
    LazyTensor::Meta meta;
    ov::Tensor allocated_tensor;
    int64_t uid;
};

void Bank::evaluate_and_allocate() {
    LOG_ERROR("Evaluating and allocating weights bank...");
    std::unique_lock guard(m_mutex);
    const char* bank_fn = "Bank::evaluate_and_allocate";

    for (auto&& bank : m_device_banks) {
        const auto& device_for_alloc = bank.first;
        auto& device_bank = bank.second;

        std::vector<LazyTensor> to_process;
        to_process.reserve(device_bank.storage.size());
        for (const auto& el : device_bank.storage) {
            // Add non-allocated tensors for furter evaluation and allocation
            if (!el.second.tensor) {
                to_process.push_back(el.second.lt);
            }
        }

        LOG_ERROR("[WEIGHT_BANK] fn=" << bank_fn << " evaluate_and_allocate() device=" << device_for_alloc
                                      << " tensors_to_process=" << to_process.size()
                                      << " already_resident=" << (device_bank.storage.size() - to_process.size()));
        const auto t0 = std::chrono::steady_clock::now();

        if (device_for_alloc == "CPU") {
            evaluate_cpu(device_bank, to_process);
        } else {
            evaluate_and_allocate_on_device(device_bank, to_process, device_for_alloc);
        }

        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        LOG_ERROR("[WEIGHT_BANK] fn=" << bank_fn << " device=" << device_for_alloc << " materialized "
                                      << to_process.size() << " tensors in " << ms
                                      << " ms (running total bytes read=" << g_bank_total_bytes.load() << " bytes ("
                                      << g_bank_total_bytes.load() / float(1024 * 1024)
                                      << " MB, tensors=" << g_bank_total_tensors.load() << " )");
    }  // for (m_device_banks)
}

void Bank::evaluate_cpu(Bank::DeviceBank& device_bank, const std::vector<LazyTensor>& to_process) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    const char* cpu_fn = "Bank::evaluate_cpu";
    ov::parallel_for(to_process.size(), [&](std::size_t idx) {
        const auto& lt = to_process[idx];
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;
        const auto t0 = std::chrono::steady_clock::now();
        auto t = lt.eval();
        device_bank.storage.at(uid).tensor = ov::Tensor(t.get_element_type(), t.get_shape());
        // Get ownership of the weights, might be a mmaped object during import
        t.copy_to(device_bank.storage.at(uid).tensor);
        const_cast<LazyTensor&>(lt).detach();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        const auto bytes = t.get_byte_size();
        g_bank_total_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_bank_total_tensors.fetch_add(1, std::memory_order_relaxed);
        if (bytes / float(1024 * 1024) > 0.1f)
            LOG_ERROR("[WEIGHT_BANK] fn=" << cpu_fn << " CPU uid=" << uid << " size_mb=" << bytes / float(1024 * 1024)
                                          << " MB eval_copy_us=" << us);
    });
}

// Note: there are no locks in this function's parallel_for
// At this point all the LazyTensor->Tensor pairs in the map are already
// allocated and there are no conflicting reads/writes since all the tensors are unique.

// FIXME: this whole flow could be improved for the same bank
// processing from different threads. We could separate LazyTensors
// evaluation and the bank access. But it requires additional rework.
void Bank::evaluate_and_allocate_on_device(Bank::DeviceBank& device_bank,
                                           const std::vector<LazyTensor>& to_process,
                                           const std::string& device) {
    // Note: not locking here. This is a private function, so Bank should handle the locks around it
    // as we lock in evaluate_and_allocate() now.
    const char* device_fn = "Bank::evaluate_and_allocate_on_device";
    std::vector<TensorToAllocate> uids_to_allocated;
    uids_to_allocated.reserve(uid_count);

    for (const auto& lt : to_process) {
        auto iter_device_registered = device_bank.registered_tensors.find(lt);
        NPUW_ASSERT(iter_device_registered != device_bank.registered_tensors.end() &&
                    "Tensor should be registered first!");
        auto uid = iter_device_registered->second;
        uids_to_allocated.push_back({lt.eval_meta(), ov::Tensor(), uid});
    }
    // Sort by UIDs, lowest first
    std::sort(uids_to_allocated.begin(),
              uids_to_allocated.end(),
              [](const TensorToAllocate& a, const TensorToAllocate& b) {
                  return a.uid < b.uid;
              });

    // Allocate memory sequentially - in order of UID
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    for (auto&& allocated : uids_to_allocated) {
        ov::SoPtr<ov::ITensor> remote_tensor =
            remote_ctx->create_host_tensor(allocated.meta.type, allocated.meta.shape);
        allocated = {allocated.meta, ov::make_tensor(remote_tensor), allocated.uid};
    }

    // Evaluate and copy into the device memory
    ov::parallel_for(uids_to_allocated.size(), [&](std::size_t idx) {
        auto& allocated = uids_to_allocated[idx];
        auto& stored_tensor = device_bank.storage.at(allocated.uid);

        const auto t0 = std::chrono::steady_clock::now();
        auto transformed = stored_tensor.lt.eval();
        transformed.copy_to(allocated.allocated_tensor);
        stored_tensor.tensor = std::move(allocated.allocated_tensor);

        // Detach the evaluated LazyTensor from its memory here - when it is 100%
        // not needed anymore (transformations, if any, and copies are done)
        // Note: this is the non-CPU path!
        const_cast<LazyTensor&>(stored_tensor.lt).detach();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        const auto bytes = transformed.get_byte_size();
        g_bank_total_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_bank_total_tensors.fetch_add(1, std::memory_order_relaxed);
        if (bytes / float(1024 * 1024) > 0.1f)
            LOG_ERROR("[WEIGHT_BANK] fn=" << device_fn << " device=" << device << " uid=" << allocated.uid
                                          << " size_mb=" << bytes / float(1024 * 1024) << " MB eval_copy_us=" << us);
    });
}

std::optional<std::string> Bank::get_name(int64_t uid) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_uid_names.find(uid);
    if (it == m_uid_names.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool Bank::is_remote(int64_t uid) const {
    LOG_ERROR("Checking if tensor is remote in weights bank: " << m_bank_name << " uid=" << uid);
    // FIXME: make generic
    std::unique_lock guard(m_mutex);

    auto npu_bank = m_device_banks.find("NPU");
    if (npu_bank != m_device_banks.end()) {
        if (npu_bank->second.storage.find(uid) != npu_bank->second.storage.end()) {
            // Found in NPU bank so considered remote (utterly wrong for the generic case)
            return true;
        }
    }
    return false;
}

void Bank::serialize(ov::npuw::orc::Stream& stream) {
    LOG_INFO("Serializing weights bank...");
    LOG_BLOCK();

    std::unique_lock guard(m_mutex);

    std::size_t bank_size = m_device_banks.size();
    stream & bank_size;

    for (const auto& elem : m_device_banks) {
        const auto& device = elem.first;
        const auto& device_bank = elem.second;
        auto storage_size = device_bank.storage.size();
        stream & device & storage_size;
        // Write tensors sequentially according to sorted uids for better memory allocation and utilization
        std::set<int64_t> uids;
        for (const auto& t_pair : device_bank.storage) {
            uids.insert(t_pair.first);
        }

        for (const auto& uid : uids) {
            stream & uid;
            auto tensor = device_bank.storage.at(uid).tensor;
            transfer_tensor(stream, tensor);
        }
    }

    LOG_INFO("DONE.");
}

void Bank::read_and_add_tensor(ov::npuw::orc::Stream& stream, int64_t uid, const std::string& device) {
    // This method is supposed to be used only during deserialization
    std::unique_lock guard(m_mutex);

    auto& device_bank = m_device_banks[device];
    auto iter_device = device_bank.storage.find(uid);

    if (iter_device != device_bank.storage.end()) {
        // Shouldn't be possible
        NPUW_ASSERT(false);
        return;
    }

    if (device == "CPU") {
        // Just read deserialized tensor into the bank
        transfer_tensor(stream, device_bank.storage[uid].tensor);
        return;
    }

    // Need to allocate on device and copy deserialized tensor to that memory
    auto remote_ctx = m_core->get_default_context(device)._ptr;
    transfer_tensor(stream,
                    device_bank.storage[uid].tensor,
                    [&remote_ctx](const ov::element::Type& type, const ov::Shape& shape) {
                        ov::SoPtr<ov::ITensor> remote_tensor = remote_ctx->create_host_tensor(type, shape);
                        return ov::make_tensor(remote_tensor);
                    });
    NPUW_ASSERT(device_bank.storage[uid].tensor && "Remote tensor should be initialized during bank deserialize");
    device_bank.storage[uid].lt = LazyTensor();
}

std::string Bank::get_name() const {
    return m_bank_name;
}

void ov::npuw::orc::serialize(Stream& stream, ov::npuw::weights::Bank& var) {
    if (stream.output()) {
        var.serialize(stream);
    } else {
        LOG_INFO("Deserializing weights bank...");
        LOG_BLOCK();

        std::size_t bank_size = 0;
        stream & bank_size;

        for (std::size_t i = 0; i < bank_size; ++i) {
            std::string device;
            stream & device;
            std::size_t storage_size = 0;
            stream & storage_size;
            for (std::size_t j = 0; j < storage_size; ++j) {
                int64_t uid = -1;
                stream & uid;
                var.read_and_add_tensor(stream, uid, device);
            }
        }

        LOG_INFO("DONE.");
    }
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name,
                                           const std::shared_ptr<const ov::ICore>& core,
                                           const std::string& alloc_device) {
    std::unique_lock guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end() || iter->second.expired()) {
        auto bank = std::make_shared<Bank>(core, alloc_device, bank_name);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core,
                                              const std::string& alloc_device) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core, alloc_device, bank_name);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core, alloc_device);
}
