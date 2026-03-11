#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "intel_npu/ops/flash_attention_tile.hpp"
#include "openvino/openvino.hpp"

std::shared_ptr<ov::Model> create_flash_attention_tile_model(const ov::Shape& q_shape,
                                                             const ov::Shape& kv_tile_shape,
                                                             const ov::Shape& mask_tile_shape,
                                                             size_t num_tiles,
                                                             const ov::element::Type& dtype = ov::element::f16) {
    using namespace ov;

    auto past_output = std::make_shared<op::v0::Parameter>(dtype, q_shape);
    auto past_max = std::make_shared<op::v0::Parameter>(dtype, Shape{q_shape[0], q_shape[1], q_shape[2]});
    auto past_sum = std::make_shared<op::v0::Parameter>(dtype, Shape{q_shape[0], q_shape[1], q_shape[2]});
    auto query = std::make_shared<op::v0::Parameter>(dtype, q_shape);

    past_output->set_friendly_name("past_output");
    past_output->get_output_tensor(0).set_names({"past_output"});
    past_max->set_friendly_name("past_max");
    past_max->get_output_tensor(0).set_names({"past_max"});
    past_sum->set_friendly_name("past_sum");
    past_sum->get_output_tensor(0).set_names({"past_sum"});
    query->set_friendly_name("Q1");
    query->get_output_tensor(0).set_names({"Q1"});

    auto mask = std::make_shared<op::v0::Parameter>(dtype, mask_tile_shape);
    mask->set_friendly_name("mask");

    std::vector<std::shared_ptr<op::v0::Parameter>> keys;
    std::vector<std::shared_ptr<op::v0::Parameter>> values;
    keys.reserve(num_tiles);
    values.reserve(num_tiles);

    for (size_t i = 0; i < num_tiles; ++i) {
        auto k = std::make_shared<op::v0::Parameter>(dtype, kv_tile_shape);
        auto v = std::make_shared<op::v0::Parameter>(dtype, kv_tile_shape);
        k->set_friendly_name("K" + std::to_string(i));
        k->get_output_tensor(0).set_names({"K" + std::to_string(i)});
        v->set_friendly_name("V" + std::to_string(i));
        v->get_output_tensor(0).set_names({"V" + std::to_string(i)});
        keys.push_back(k);
        values.push_back(v);
    }

    Output<Node> acc = past_output;
    Output<Node> mx = past_max;
    Output<Node> sm = past_sum;

    for (size_t i = 0; i < num_tiles; ++i) {
        ov::intel_npu::op::FlashAttentionTile::Config cfg;
        cfg.is_head = (i == 0);
        cfg.is_tail = (i == num_tiles - 1);

        auto fa =
            std::make_shared<ov::intel_npu::op::FlashAttentionTile>(query, keys[i], values[i], acc, mx, sm, mask, cfg);
        fa->set_friendly_name("flash_attention_tile_" + std::to_string(i));
        acc = fa->output(0);
        mx = fa->output(1);
        sm = fa->output(2);
    }

    ResultVector results{std::make_shared<op::v0::Result>(acc),
                         std::make_shared<op::v0::Result>(mx),
                         std::make_shared<op::v0::Result>(sm)};

    ParameterVector params{past_output, past_max, past_sum, query};
    for (size_t i = 0; i < num_tiles; ++i) {
        params.push_back(keys[i]);
        params.push_back(values[i]);
    }
    params.push_back(mask);

    return std::make_shared<Model>(results, params, "FlashAttention" + std::to_string(num_tiles) + "Tiles");
}

static ov::Tensor make_random_tensor(const ov::element::Type& dtype, const ov::Shape& shape) {
    ov::Tensor tensor(dtype, shape);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(dist(rng));
    }
    return tensor;
}

auto compute_median = [](std::vector<double> times) -> double {
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n % 2 == 0)
        return (times[n / 2 - 1] + times[n / 2]) / 2.0;
    return times[n / 2];
};

auto compute_mean_skip_first = [](const std::vector<double>& times, size_t skip = 10) -> double {
    if (times.size() <= skip)
        return 0.0;
    double sum = 0.0;
    for (size_t i = skip; i < times.size(); ++i)
        sum += times[i];
    return sum / (times.size() - skip);
};

int main(int argc, char** argv) {
    const std::string device = "NPU";
    const size_t num_iters = 200;

    constexpr size_t BATCH = 1;
    constexpr size_t SEQ_LEN = 1024;
    constexpr size_t HEAD_DIM = 128;
    constexpr size_t HEAD_NUM = 32;

    const ov::Shape q_shape{BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    const ov::Shape kv_tile_shape{BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    const ov::Shape mask_tile_shape{BATCH, 1, SEQ_LEN, SEQ_LEN};
    const ov::Shape max_sum_shape{BATCH, HEAD_NUM, SEQ_LEN};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto fill_random_f16 = [&](ov::Tensor t) {
        auto* data = t.data<ov::float16>();
        for (size_t i = 0; i < t.get_size(); ++i)
            data[i] = ov::float16(dist(rng));
    };
    ov::Core core;
    // core.set_property(ov::log::level(ov::log::Level::TRACE));

    // 8 Nodes
    if (1) {
        std::cout << "[8-node model] Building model" << std::endl;
        auto model_8 = create_flash_attention_tile_model(q_shape, kv_tile_shape, mask_tile_shape, 8);
        auto compiled_8 = core.compile_model(model_8, device);
        auto infer_req_8 = compiled_8.create_infer_request();
        std::cout << "[8-node model] Filling tensors" << std::endl;
        for (int i = 0; i < compiled_8.inputs().size(); ++i) {
            fill_random_f16(infer_req_8.get_input_tensor(i));
        }

        std::cout << "[8-node model] Running  model for " << num_iters << " iterations" << std::endl;

        std::vector<double> iter_times;
        iter_times.reserve(num_iters * 2);
        for (size_t iter = 0; iter < num_iters; ++iter) {
            const auto t0 = std::chrono::steady_clock::now();
            infer_req_8.infer();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            iter_times.push_back(ms);
        }

        std::cout << "[8-node model] Mean (skip 10) " << compute_mean_skip_first(iter_times) << " ms."
                  << " Mean time per node " << compute_mean_skip_first(iter_times) / 8 << " ms."
                  << " Median " << compute_median(iter_times) << " ms."
                  << " Median time per node " << compute_median(iter_times) / 8 << " ms." << std::endl;

        if (1) {
            std::cout << "\nIteration times (ms): ";
            for (const auto& t : iter_times) {
                std::cout << t << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "\n----------------------------------------\n" << std::endl;
    if (1) {
        // 1 Node
        std::cout << "[1-node model] Building 1-tile model" << std::endl;
        auto model_1 = create_flash_attention_tile_model(q_shape, kv_tile_shape, mask_tile_shape, 1, ov::element::f16);
        auto compiled_1 = core.compile_model(model_1, device);
        auto infer_req_1 = compiled_1.create_infer_request();

        std::cout << "[1-node model] Filling 1-tile tensors" << std::endl;
        for (int i = 0; i < model_1->inputs().size(); ++i) {
            fill_random_f16(infer_req_1.get_input_tensor(i));
        }

        std::cout << "[1-node model] Running model for " << num_iters << " iterations" << std::endl;

        std::vector<double> iter_times;
        iter_times.reserve(num_iters * 2);
        for (size_t iter = 0; iter < num_iters; ++iter) {
            const auto t0 = std::chrono::steady_clock::now();
            infer_req_1.infer();
            for (size_t tile = 0; tile < 7; ++tile) {
                auto past_output_t = infer_req_1.get_output_tensor(0);
                auto past_max_t = infer_req_1.get_output_tensor(1);
                auto past_sum_t = infer_req_1.get_output_tensor(2);
                infer_req_1.set_tensor("past_output", past_output_t);
                infer_req_1.set_tensor("past_max", past_max_t);
                infer_req_1.set_tensor("past_sum", past_sum_t);
                infer_req_1.infer();
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            iter_times.push_back(ms);
        }

        std::cout << "[1-node model] Mean (skip 10) " << compute_mean_skip_first(iter_times) << " ms."
                  << " Mean time per node " << compute_mean_skip_first(iter_times) / 8 << " ms."
                  << " Median " << compute_median(iter_times) << " ms."
                  << " Median time per node " << compute_median(iter_times) / 8 << " ms." << std::endl;

        if (1) {
            std::cout << "\nIteration times (ms): ";

            for (const auto& t : iter_times) {
                std::cout << t << " ";
            }
            std::cout << std::endl;
        }
    }
    // std::cout << "\nAll iterations completed." << std::endl;
    return 0;
}