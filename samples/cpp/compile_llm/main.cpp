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
    past_max->set_friendly_name("past_max");
    past_sum->set_friendly_name("past_sum");
    query->set_friendly_name("Q1");

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
        v->set_friendly_name("V" + std::to_string(i));
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

static ov::Tensor make_filled_tensor(const ov::element::Type& dtype, const ov::Shape& shape, float val) {
    ov::Tensor tensor(dtype, shape);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(val);
    }
    return tensor;
}

int main(int argc, char** argv) {
    const std::string device = "NPU";
    const size_t num_iters = 100;

    constexpr size_t BATCH = 1;
    constexpr size_t SEQ_LEN = 1024;
    constexpr size_t HEAD_DIM = 128;
    constexpr size_t HEAD_NUM = 32;

    const ov::Shape q_shape{BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    const ov::Shape kv_tile_shape{BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    const ov::Shape mask_tile_shape{BATCH, 1, SEQ_LEN, SEQ_LEN};
    const ov::Shape max_sum_shape{BATCH, HEAD_NUM, SEQ_LEN};

    ov::Core core;

    // --- Model A: 8 tiles fused ---
    std::cout << "Building 8-tile model..." << std::endl;
    auto model_8 = create_flash_attention_tile_model(q_shape, kv_tile_shape, mask_tile_shape, 8, ov::element::f16);
    auto compiled_8 = core.compile_model(model_8, device);
    auto infer_req_8 = compiled_8.create_infer_request();

    double total_ms_8 = 0.0;
    std::cout << "Running 8-tile model for " << num_iters << " iteration(s)..." << std::endl;
    for (size_t iter = 0; iter < num_iters; ++iter) {
        infer_req_8.set_tensor("past_output", make_random_tensor(ov::element::f16, q_shape));
        infer_req_8.set_tensor("past_max", make_random_tensor(ov::element::f16, max_sum_shape));
        infer_req_8.set_tensor("past_sum", make_random_tensor(ov::element::f16, max_sum_shape));
        infer_req_8.set_tensor("Q1", make_random_tensor(ov::element::f16, q_shape));
        infer_req_8.set_tensor("mask", make_random_tensor(ov::element::f16, mask_tile_shape));
        for (size_t t = 0; t < 8; ++t) {
            infer_req_8.set_tensor("K" + std::to_string(t), make_random_tensor(ov::element::f16, kv_tile_shape));
            infer_req_8.set_tensor("V" + std::to_string(t), make_random_tensor(ov::element::f16, kv_tile_shape));
        }

        const auto t0 = std::chrono::steady_clock::now();
        infer_req_8.infer();
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms_8 += ms;
        std::cout << "[8-tile] iter " << iter << " infer=" << ms << " ms" << std::endl;
    }
    std::cout << "[8-tile] avg infer=" << total_ms_8 / num_iters << " ms over " << num_iters << " iters" << std::endl;

    // --- Model B: 1 tile, run 8 times chaining past_ outputs ---
    std::cout << "\nBuilding 1-tile model..." << std::endl;
    auto model_1 = create_flash_attention_tile_model(q_shape, kv_tile_shape, mask_tile_shape, 1, ov::element::f16);
    auto compiled_1 = core.compile_model(model_1, device);
    auto infer_req_1 = compiled_1.create_infer_request();

    const auto q_tensor = make_random_tensor(ov::element::f16, q_shape);
    const auto mask_tensor = make_random_tensor(ov::element::f16, mask_tile_shape);

    double total_ms_1 = 0.0;
    std::cout << "Running 1-tile model in loop of 8 for " << num_iters << " outer iteration(s)..." << std::endl;
    for (size_t iter = 0; iter < num_iters; ++iter) {
        ov::Tensor past_output_t = make_filled_tensor(ov::element::f16, q_shape, 0.f);
        ov::Tensor past_max_t = make_filled_tensor(ov::element::f16, max_sum_shape, -65504.f);
        ov::Tensor past_sum_t = make_filled_tensor(ov::element::f16, max_sum_shape, 0.f);

        infer_req_1.set_tensor("Q1", q_tensor);
        infer_req_1.set_tensor("mask", mask_tensor);

        const auto t0 = std::chrono::steady_clock::now();
        infer_req_1.set_tensor("K0", make_random_tensor(ov::element::f16, kv_tile_shape));
        infer_req_1.set_tensor("V0", make_random_tensor(ov::element::f16, kv_tile_shape));
        for (size_t tile = 0; tile < 8; ++tile) {
            infer_req_1.set_tensor("past_output", past_output_t);
            infer_req_1.set_tensor("past_max", past_max_t);
            infer_req_1.set_tensor("past_sum", past_sum_t);
            infer_req_1.infer();
            past_output_t = infer_req_1.get_output_tensor(0);
            past_max_t = infer_req_1.get_output_tensor(1);
            past_sum_t = infer_req_1.get_output_tensor(2);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms_1 += ms;
    }
    std::cout << "[1-tile loop] avg total_infer=" << total_ms_1 / num_iters << " ms over " << num_iters << " iters"
              << std::endl;

    std::cout << "\nAll iterations completed." << std::endl;
    return 0;
}