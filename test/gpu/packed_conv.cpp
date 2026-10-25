/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/gpu/packed_conv.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/half.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/json.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <test.hpp>
#include <cmath>

namespace {
using migraphx::half;
using migraphx::shape;

void make_head(migraphx::module& m,
               std::size_t ic,
               bool pointwise,
               shape::type_t type = shape::half_type,
               std::size_t width  = 10,
               bool extra_output  = false,
               bool wrong_sigmoid = false)
{
    auto x = m.add_parameter("x0", shape{type, {1, ic, width, width}});
    // Deliberately insert bias before weights; kernel argument order must follow semantics.
    auto b    = m.add_parameter("x2", shape{type, {32}});
    auto w    = m.add_parameter("x1", shape{type, {32, ic, 3, 3}});
    auto conv = m.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
    auto bias = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}), b);
    auto sum        = m.add_instruction(migraphx::make_op("add"), conv, bias);
    auto sig        = m.add_instruction(migraphx::make_op("sigmoid"), wrong_sigmoid ? conv : sum);
    auto activation = m.add_instruction(migraphx::make_op("mul"), sig, sum);
    auto result     = activation;
    if(pointwise)
    {
        auto pw        = m.add_parameter("x3", shape{type, {4, 32, 1, 1}});
        auto pb        = m.add_parameter("x4", shape{type, {4}});
        auto last      = m.add_instruction(migraphx::make_op("convolution"), activation, pw);
        auto last_bias = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", last->get_shape().lens()}}),
            pb);
        result = m.add_instruction(migraphx::make_op("add"), last_bias, last);
        result = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, -1}}}), result);
    }
    if(extra_output)
        m.add_return({result, activation});
    else
        m.add_return({result});
}

std::vector<shape> input_shapes(const migraphx::module& m)
{
    auto params = m.get_parameters();
    migraphx::sort_params(params);
    return migraphx::to_shapes(params);
}

// The host transport half type truncates conversions. Use an independent
// native IEEE half conversion for the GPU's round-to-nearest-even boundary.
half nearest_half(double value) { return migraphx::bit_cast<half>(static_cast<_Float16>(value)); }

std::vector<half> samples(std::size_t size, int seed)
{
    std::vector<half> result(size);
    for(std::size_t i = 0; i < size; ++i)
        result[i] = float(int((i * 17 + seed) % 31) - 15) / 64;
    return result;
}

std::vector<double>
reference_head(const std::vector<std::vector<half>>& data, std::size_t ic, bool pointwise)
{
    std::vector<half> intermediate(3200);
    std::vector<double> result(pointwise ? 400 : 3200);
    for(std::size_t c = 0; c < 32; ++c)
        for(std::size_t h = 0; h < 10; ++h)
            for(std::size_t w = 0; w < 10; ++w)
            {
                double sum = double(data[2][c]);
                for(std::size_t k = 0; k < ic; ++k)
                    for(int y = 0; y < 3; ++y)
                        for(int x = 0; x < 3; ++x)
                        {
                            const int ih = int(h) + y - 1, iw = int(w) + x - 1;
                            if(ih >= 0 and ih < 10 and iw >= 0 and iw < 10)
                                sum += double(data[0][k * 100 + ih * 10 + iw]) *
                                       double(data[1][(c * ic + k) * 9 + y * 3 + x]);
                        }
                const auto index = c * 100 + h * 10 + w;
                if(not pointwise)
                    result[index] = sum / (1 + std::exp(-sum));
                intermediate[index] = nearest_half(sum / (1 + std::exp(-sum)));
            }
    if(pointwise)
        for(std::size_t c = 0; c < 4; ++c)
            for(std::size_t n = 0; n < 100; ++n)
            {
                double sum = double(data[4][c]);
                for(std::size_t k = 0; k < 32; ++k)
                    sum += double(intermediate[k * 100 + n]) * double(data[3][c * 32 + k]);
                result[c * 100 + n] = sum;
            }
    return result;
}

void check_native(std::size_t ic, bool pointwise, bool permuted = false, bool paired = false)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx) or
       (paired and not migraphx::gpu::mlir_enabled()))
    {
        std::cout << "Skipping packed convolution execution on unsupported GPU" << std::endl;
        return;
    }
    const auto build = [&](migraphx::module& m) {
        make_head(m, ic, pointwise);
        if(permuted)
        {
            m.rename_parameter(m.get_parameter("x0"), "z");
            m.rename_parameter(m.get_parameter("x1"), "a");
            m.rename_parameter(m.get_parameter("x2"), "m");
        }
    };
    migraphx::program original;
    build(*original.get_main_module());
    auto shapes  = input_shapes(*original.get_main_module());
    auto problem = migraphx::gpu::analyze_packed_conv(*original.get_main_module(), shapes);
    EXPECT(bool(problem));
    if(not problem)
        return;
    std::vector<std::vector<half>> data;
    for(std::size_t i = 0; i < shapes.size(); ++i)
        data.push_back(samples(shapes[i].elements(), int(i)));
    // Use the fixture's known semantic names, not the analyzer's argument map.
    // Renamed parameters sort as weights,bias,[pointwise weights,bias],image.
    auto semantic = data;
    if(permuted)
        semantic = pointwise
                       ? std::vector<std::vector<half>>{data[4], data[0], data[1], data[2], data[3]}
                       : std::vector<std::vector<half>>{data[2], data[0], data[1]};
    const auto expected = reference_head(semantic, ic, pointwise);
    auto solutions      = migraphx::gpu::packed_conv_solutions(*problem, true);
    if(paired)
        solutions.push_back(migraphx::value{}); // ordinary first and second stages
    for(const auto& solution : solutions)
    {
        migraphx::program p;
        auto* body = p.create_module("head");
        build(*body);
        auto* mm = p.get_main_module();
        std::vector<migraphx::instruction_ref> args;
        std::unordered_map<std::string, migraphx::argument> parameters;
        std::vector<migraphx::argument> owners;
        for(std::size_t i = 0; i < shapes.size(); ++i)
        {
            const auto name = "x" + std::to_string(i);
            args.push_back(mm->add_parameter(name, shapes[i]));
            // Wrong argument order must cause a numeric failure, not a memory fault.
            auto storage = data[i];
            if(permuted)
                storage.resize(32 * ic * 9, half(0));
            const shape allocation{shape::half_type, {storage.size()}};
            owners.push_back(migraphx::gpu::to_gpu(migraphx::argument{allocation, storage.data()}));
            parameters[name] = migraphx::argument{shapes[i], owners.back().data()};
        }
        const auto output = body->get_output_shapes().front();
        args.push_back(mm->add_parameter("output", output));
        const shape guarded_shape{shape::half_type, {output.elements() + 16}};
        std::vector<half> canaries(guarded_shape.elements(), half(1234));
        const auto guarded =
            migraphx::gpu::to_gpu(migraphx::argument{guarded_shape, canaries.data()});
        parameters["output"] = migraphx::argument{output, guarded.data()};
        auto ins             = mm->add_instruction(
            migraphx::make_op("gpu::precompile_op", {{"output_shape", migraphx::to_value(output)}}),
            args,
            {body});
        mm->add_return({ins});
        const auto saved = migraphx::from_json_string(migraphx::to_json_string(solution));
        if(paired)
        {
            const migraphx::value configuration{
                {"provider", "rocmlir_pair"}, {"first", saved}, {"second", migraphx::value{}}};
            migraphx::gpu::compile(
                ctx, ins, migraphx::make_op("gpu::mlir_op", {{"fast_math", true}}), configuration)
                .replace(*mm, ins);
        }
        else
        {
            auto kernel =
                migraphx::gpu::compile_packed_conv(ctx, *body, migraphx::to_shapes(args), saved);
            mm->replace_instruction(ins, kernel, args);
        }
        p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});
        for(int repeat = 0; repeat < 3; ++repeat)
        {
            const auto actual = migraphx::gpu::from_gpu(p.eval(parameters).front());
            actual.visit([&](auto values) {
                EXPECT(values.size() == expected.size());
                for(std::size_t i = 0; i < expected.size(); ++i)
                    EXPECT(std::isfinite(double(values[i])) and
                           std::abs(double(values[i]) - expected[i]) <=
                               0.01 + 0.005 * std::abs(expected[i]));
            });
        }
        migraphx::gpu::from_gpu(guarded).visit([&](auto values) {
            for(std::size_t i = output.elements(); i < values.size(); ++i)
                EXPECT(double(values[i]) == 1234);
        });
        std::cout << "Validated " << solution << std::endl;
    }
}
} // namespace

TEST_CASE(packed_conv_analysis)
{
    for(const auto ic : {32u, 512u})
    {
        migraphx::module m;
        make_head(m, ic, false);
        EXPECT(bool(migraphx::gpu::analyze_packed_conv(m, input_shapes(m))));
    }
    migraphx::module fused;
    make_head(fused, 32, true);
    EXPECT(bool(migraphx::gpu::analyze_packed_conv(fused, input_shapes(fused))));
}

TEST_CASE(packed_conv_rejects_unsupported_graphs)
{
    migraphx::module wrong_type, wrong_width, width_640, wrong_channels, extra_output,
        wrong_sigmoid;
    make_head(wrong_type, 32, false, shape::float_type);
    make_head(wrong_width, 32, false, shape::half_type, 9);
    make_head(width_640, 512, false, shape::half_type, 20);
    make_head(wrong_channels, 64, false);
    make_head(extra_output, 32, true, shape::half_type, 10, true);
    make_head(wrong_sigmoid, 32, false, shape::half_type, 10, false, true);
    for(const auto* m :
        {&wrong_type, &wrong_width, &width_640, &wrong_channels, &extra_output, &wrong_sigmoid})
        EXPECT(not migraphx::gpu::analyze_packed_conv(*m, input_shapes(*m)));
}

TEST_CASE(packed_conv_rejects_invalid_launch)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx))
        return;
    migraphx::module m;
    make_head(m, 32, true);
    auto shapes  = input_shapes(m);
    const auto p = migraphx::gpu::analyze_packed_conv(m, shapes);
    EXPECT(bool(p));
    if(not p)
        return;
    auto solution = migraphx::gpu::packed_conv_solutions(*p, false).front();
    EXPECT(test::throws([&] { migraphx::gpu::compile_packed_conv(ctx, m, shapes, solution); }));
    shapes.push_back(m.get_output_shapes().front());
    solution["split"] = 5;
    EXPECT(test::throws([&] { migraphx::gpu::compile_packed_conv(ctx, m, shapes, solution); }));
}

TEST_CASE(packed_conv_search_cache_scope)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx) or not migraphx::gpu::mlir_enabled())
        return;
    migraphx::program p;
    auto* body = p.create_module("head");
    make_head(*body, 512, false);
    auto* mm          = p.get_main_module();
    const auto shapes = input_shapes(*body);
    std::vector<migraphx::instruction_ref> args;
    for(std::size_t i = 0; i < shapes.size(); ++i)
        args.push_back(mm->add_parameter("x" + std::to_string(i), shapes[i]));
    const auto output = body->get_output_shapes().front();
    args.push_back(mm->add_parameter("output", output));
    const auto ins = mm->add_instruction(
        migraphx::make_op("gpu::precompile_op", {{"output_shape", migraphx::to_value(output)}}),
        args,
        {body});
    const auto quick = migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, false, true);
    const auto full  = migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, true, true);
    EXPECT(bool(quick) and bool(full));
    if(not quick or not full)
        return;
    ctx.get_problem_cache().insert("gpu::mlir_op", quick->problem, quick->solutions.back());
    EXPECT(ctx.get_problem_cache()
               .get("gpu::mlir_op",
                    migraphx::from_json_string(migraphx::to_json_string(quick->problem)))
               .has_value());
    EXPECT(not ctx.get_problem_cache().get("gpu::mlir_op", full->problem));
    EXPECT(full->solutions.size() > quick->solutions.size());
}

TEST_CASE(packed_conv_region_matching)
{
    migraphx::gpu::context gpu_ctx;
    if(not migraphx::gpu::packed_conv_supported(gpu_ctx) or not migraphx::gpu::mlir_enabled())
        return;
    for(bool shared : {false, true})
        for(bool fast_math : {false, true})
        {
            migraphx::program p;
            make_head(*p.get_main_module(), 32, true, shape::half_type, 10, shared);
            migraphx::context ctx = gpu_ctx;
            bool reached_fusion   = false;
            for(const auto& pass :
                migraphx::gpu::target{}.get_passes(ctx, {.fast_math = fast_math}))
            {
                migraphx::run_passes(p, {pass});
                if(pass.name() != "gpu::fuse_mlir")
                    continue;
                reached_fusion          = true;
                std::size_t fused_heads = 0;
                for(auto ins : migraphx::iterator_for(*p.get_main_module()))
                {
                    if(ins->name() != "gpu::mlir_op" or ins->module_inputs().size() != 1)
                        continue;
                    const auto problem = migraphx::gpu::analyze_packed_conv(
                        *ins->module_inputs().front(), migraphx::to_shapes(ins->inputs()));
                    if(problem and problem->pointwise)
                        ++fused_heads;
                }
                EXPECT(fused_heads == (shared or not fast_math ? 0u : 1u));
                break;
            }
            EXPECT(reached_fusion);
        }
}

TEST_CASE(packed_conv_stage_cache_dependency)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx) or not migraphx::gpu::mlir_enabled())
        return;
    migraphx::program p;
    auto* body = p.create_module("head");
    make_head(*body, 32, true);
    auto* mm    = p.get_main_module();
    auto shapes = input_shapes(*body);
    shapes.push_back(body->get_output_shapes().front());
    std::vector<migraphx::instruction_ref> args;
    for(std::size_t i = 0; i < shapes.size(); ++i)
        args.push_back(mm->add_parameter("x" + std::to_string(i), shapes[i]));
    auto ins = mm->add_instruction(
        migraphx::make_op("gpu::precompile_op",
                          {{"output_shape", migraphx::to_value(shapes.back())}}),
        args,
        {body});
    const auto initial = migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, false, true);
    EXPECT(bool(initial));
    if(not initial)
        return;
    const auto key       = migraphx::from_json_string(initial->problem.to<std::string>());
    const auto stages    = key.at("base").at("problem");
    const auto first_key = stages.at(0).at("problem");
    EXPECT(ctx.get_problem_cache().get("gpu::mlir_op", first_key).has_value());
    // A changed ordinary-stage winner must invalidate a previously cached region.
    const auto previous = stages.at(0).at("config");
    const migraphx::value alternate =
        previous.is_null() ? migraphx::value{"v3:64,32,32,4,2,2,1,1,2"} : migraphx::value{};
    ctx.get_problem_cache().insert(
        "gpu::mlir_op", first_key, migraphx::value{{"config", alternate}});
    const auto updated = migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, false, true);
    EXPECT(bool(updated));
    if(updated)
        EXPECT(initial->problem != updated->problem);
}

TEST_CASE(packed_conv_fp32_fusion_contract)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx))
        return;
    migraphx::module body;
    make_head(body, 512, false);
    auto shapes = input_shapes(body);
    shapes.push_back(body.get_output_shapes().front());
    auto problem = migraphx::gpu::analyze_packed_conv(body, shapes);
    if(not problem)
        return;
    auto kernel = migraphx::gpu::compile_packed_conv(
        ctx, body, shapes, migraphx::gpu::packed_conv_solutions(*problem, false).front());
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::instruction_ref> args;
    std::unordered_map<std::string, migraphx::argument> parameters;
    for(std::size_t i = 0; i < shapes.size(); ++i)
    {
        const auto name = "x" + std::to_string(i);
        args.push_back(mm->add_parameter(name, shapes[i]));
        std::vector<half> data(shapes[i].elements(), half(0));
        if(i == 0)
            std::fill_n(data.begin(), 200, half(1));
        if(i == 1)
            for(std::size_t c = 0; c < 32; ++c)
                for(std::size_t k = 0; k < 2; ++k)
                    data[(c * 512 + k) * 9 + 4] = half(35008);
        if(i == 2)
            std::fill(data.begin(), data.end(), half(-65504));
        parameters[name] = migraphx::gpu::to_gpu(migraphx::argument{shapes[i], data.data()});
    }
    mm->add_return({mm->add_instruction(kernel, args)});
    p.compile(migraphx::make_target("gpu"));
    const auto result = migraphx::gpu::from_gpu(p.eval(parameters).front());
    result.visit([&](auto values) {
        for(auto x : values)
            EXPECT(float(x) == 4512.f);
    });
    // The explicit FP16 convolution would overflow before the bias cancellation.
    EXPECT(std::isinf(float(nearest_half(35008.0 + 35008.0))));
}

TEST_CASE(packed_conv_extreme_numerics)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx))
        return;
    // Sparse center taps isolate half subnormal inputs, signed near-zero SiLU,
    // rounding boundaries and large products without a permissive absolute floor.
    const double tiny                              = std::ldexp(1.0, -24);
    const std::vector<std::array<double, 3>> cases = {{tiny, 65504, 0},
                                                      {65504, tiny, 0},
                                                      {-tiny, 65504, 0},
                                                      {0, 0, -20},
                                                      {0, 0, -18},
                                                      {0, 0, 0.3},
                                                      {1, 1, std::ldexp(1.0, -11)},
                                                      {1, -1, std::ldexp(1.0, -11)}};
    for(bool pointwise : {false, true})
    {
        const std::size_t ic = pointwise ? 32 : 512;
        migraphx::module body;
        make_head(body, ic, pointwise);
        auto shapes = input_shapes(body);
        shapes.push_back(body.get_output_shapes().front());
        const auto problem = migraphx::gpu::analyze_packed_conv(body, shapes);
        EXPECT(bool(problem));
        if(not problem)
            return;
        auto solutions = migraphx::gpu::packed_conv_solutions(*problem, false);
        solutions.push_back(migraphx::gpu::packed_conv_solutions(*problem, true).front());
        for(const auto& solution : solutions)
        {
            auto kernel = migraphx::gpu::compile_packed_conv(ctx, body, shapes, solution);
            migraphx::program p;
            auto* mm = p.get_main_module();
            std::vector<migraphx::instruction_ref> args;
            for(std::size_t i = 0; i < shapes.size(); ++i)
                args.push_back(mm->add_parameter("x" + std::to_string(i), shapes[i]));
            mm->add_return({mm->add_instruction(kernel, args)});
            p.compile(migraphx::make_target("gpu"));
            for(const auto& values : cases)
            {
                std::unordered_map<std::string, migraphx::argument> parameters;
                for(std::size_t i = 0; i < shapes.size(); ++i)
                {
                    std::vector<half> data(shapes[i].elements(), half(0));
                    if(i == 0)
                        std::fill_n(data.begin(), 100, nearest_half(values[0]));
                    if(i == 1)
                        for(std::size_t c = 0; c < 32; ++c)
                            data[c * ic * 9 + 4] = nearest_half(values[1]);
                    if(i == 2)
                        std::fill(data.begin(), data.end(), nearest_half(values[2]));
                    if(pointwise and i == 3)
                        for(std::size_t c = 0; c < 4; ++c)
                            data[c * 32 + c] = half(1);
                    parameters["x" + std::to_string(i)] =
                        migraphx::gpu::to_gpu(migraphx::argument{shapes[i], data.data()});
                }
                const double sum =
                    double(nearest_half(values[0])) * double(nearest_half(values[1])) +
                    double(nearest_half(values[2]));
                const double expected = double(nearest_half(sum / (1 + std::exp(-sum))));
                const auto result     = migraphx::gpu::from_gpu(p.eval(parameters).front());
                result.visit([&](auto actual) {
                    for(auto x : actual)
                        EXPECT(std::isfinite(double(x)) and
                               std::abs(double(x) - expected) <= tiny + 0.001 * std::abs(expected));
                });
            }
        }
    }
}

TEST_CASE(packed_conv_nearest_even_oracle)
{
    EXPECT(float(nearest_half(1 + std::ldexp(1.0, -11))) == 1);
    EXPECT(float(nearest_half(1 + 3 * std::ldexp(1.0, -11))) == 1 + std::ldexp(1.0, -9));
    EXPECT(float(nearest_half(std::ldexp(1.0, -25))) == 0);
    EXPECT(float(nearest_half(3 * std::ldexp(1.0, -25))) == std::ldexp(1.0, -23));
    EXPECT(std::isinf(float(nearest_half(70016))));
}

TEST_CASE(packed_conv_math_permission)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx))
        return;
    migraphx::program p;
    auto* body = p.create_module("head");
    make_head(*body, 32, false);
    auto* mm    = p.get_main_module();
    auto shapes = input_shapes(*body);
    shapes.push_back(body->get_output_shapes().front());
    std::vector<migraphx::instruction_ref> args;
    for(std::size_t i = 0; i < shapes.size(); ++i)
        args.push_back(mm->add_parameter("x" + std::to_string(i), shapes[i]));
    auto ins = mm->add_instruction(
        migraphx::make_op("gpu::precompile_op",
                          {{"output_shape", migraphx::to_value(shapes.back())}}),
        args,
        {body});
    EXPECT(not migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, false, false));
    if(not migraphx::gpu::mlir_enabled())
    {
        EXPECT(not migraphx::gpu::get_packed_conv_tuning_config(ctx, ins, false, true));
        return;
    }
    const auto problem = migraphx::gpu::analyze_packed_conv(*body, shapes);
    EXPECT(bool(problem));
    if(not problem)
        return;
    const auto solution = migraphx::gpu::packed_conv_solutions(*problem, false).front();
    EXPECT(test::throws(
        [&] { migraphx::gpu::compile(ctx, ins, migraphx::make_op("gpu::mlir_op"), solution); }));
}

TEST_CASE(packed_conv_stage_prefills)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx) or not migraphx::gpu::mlir_enabled())
        return;
    migraphx::module body;
    make_head(body, 32, false);
    auto shapes = input_shapes(body);
    shapes.push_back(body.get_output_shapes().front());
    const auto problem = migraphx::gpu::analyze_packed_conv(body, shapes);
    if(not problem)
        return;
    auto op = migraphx::gpu::compile_packed_conv(
        ctx, body, shapes, migraphx::gpu::packed_conv_solutions(*problem, false).front());
    migraphx::gpu::mlir_code_object compiled{migraphx::any_cast<migraphx::gpu::code_object_op>(op)};
    compiled.prefill_indices = {2, 3};
    compiled.prefill_values  = {0, 0};
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::instruction_ref> args;
    std::unordered_map<std::string, migraphx::argument> parameters;
    for(std::size_t i = 0; i < shapes.size(); ++i)
    {
        const auto name = "x" + std::to_string(i);
        args.push_back(mm->add_parameter(name, shapes[i]));
        std::vector<half> zero(shapes[i].elements(), half(i == 2 ? 7 : i == 3 ? 11 : 0));
        parameters[name] = migraphx::gpu::to_gpu(migraphx::argument{shapes[i], zero.data()});
    }
    auto ins    = mm->add_instruction(migraphx::make_op("identity"), args.back());
    auto result = migraphx::gpu::insert_mlir(*mm, ins, compiled, args);
    mm->add_return({result});
    mm->remove_instruction(ins);
    p.compile(migraphx::make_target("gpu"));
    const auto output = migraphx::gpu::from_gpu(p.eval(parameters).front());
    output.visit([&](auto values) {
        for(auto x : values)
            EXPECT(float(x) == 0.f);
    });
    migraphx::gpu::from_gpu(parameters.at("x2")).visit([&](auto values) {
        for(auto x : values)
            EXPECT(float(x) == 0.f);
    });
}

TEST_CASE(packed_conv_tuple_prefills)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::mlir_enabled())
        return;
    const shape tensor{shape::float_type, {4}};
    const shape outputs{{tensor, tensor}};
    migraphx::gpu::hip_compile_options options;
    options.inputs = {tensor, tensor, tensor};
    options.output = outputs;
    options.local = options.global = 64;
    options.kernel_name            = "prefill_pair";
    const auto kernel              = migraphx::gpu::compile_hip_code_object(ctx,
                                                                            R"__migraphx__(
extern "C" __global__ void prefill_pair(const float* x, float* left, float* right)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 4) return;
    left[i] += x[i] + 1;
    right[i] += x[i] + 2;
}
)__migraphx__",
                                                                            options);
    migraphx::gpu::mlir_code_object compiled{
        migraphx::any_cast<migraphx::gpu::code_object_op>(kernel)};
    compiled.prefill_indices = {0, 1, 2};
    compiled.prefill_values  = {0, 0, 0};
    migraphx::program p;
    auto* mm         = p.get_main_module();
    auto input       = mm->add_parameter("input", tensor);
    auto output      = mm->add_parameter("output", outputs);
    auto placeholder = mm->add_instruction(migraphx::make_op("identity"), output);
    auto result      = migraphx::gpu::insert_mlir(*mm, placeholder, compiled, {input, output});
    mm->add_return({result});
    mm->remove_instruction(placeholder);
    // Match the final compiler phase for this already compiled GPU graph.
    migraphx::run_passes(*mm, {migraphx::eliminate_identity{}});
    std::vector<migraphx::context> contexts{ctx};
    mm->finalize(contexts);
    std::vector<float> initial(4, 7);
    const auto buffer = [&] {
        return migraphx::gpu::to_gpu(migraphx::argument{tensor, initial.data()});
    };
    const auto source = buffer();
    const migraphx::argument tuple({buffer(), buffer()});
    for(int repeat = 0; repeat < 3; ++repeat)
    {
        const auto actual =
            migraphx::gpu::from_gpu(
                p.eval_with_context(contexts, {{"input", source}, {"output", tuple}}).front())
                .get_sub_objects();
        EXPECT(actual.size() == 2);
        for(std::size_t i = 0; i < actual.size(); ++i)
            actual[i].visit([&](auto values) {
                for(auto value : values)
                    EXPECT(float(value) == float(i + 1));
            });
    }
}

TEST_CASE(packed_conv_packed_arithmetic)
{
    migraphx::gpu::context ctx;
    if(not migraphx::gpu::packed_conv_supported(ctx))
        return;
    const std::vector<uint16_t> bits{
        0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x3555, 0xb555};
    std::vector<uint32_t> left, right;
    std::vector<float> sums, expected;
    for(std::size_t a = 0; a < bits.size(); ++a)
        for(std::size_t b = 0; b < bits.size(); ++b)
            for(float sum : {0.f, 1.f, -1.f, 1024.f, -1024.f, 1e-15f})
            {
                const auto x = uint32_t(bits[a]) | (uint32_t(bits[(a + 1) % bits.size()]) << 16);
                const auto y = uint32_t(bits[b]) | (uint32_t(bits[(b + 5) % bits.size()]) << 16);
                left.push_back(x);
                right.push_back(y);
                sums.push_back(sum);
                for(auto i : {0, 1})
                    for(auto j : {0, 1})
                        expected.push_back(
                            std::fma(float(migraphx::bit_cast<half>(uint16_t(x >> (16 * i)))),
                                     float(migraphx::bit_cast<half>(uint16_t(y >> (16 * j)))),
                                     sum));
            }
    const shape packed{shape::int32_type, {left.size()}};
    const shape accum{shape::float_type, {sums.size()}};
    const shape output{shape::float_type, {expected.size()}};
    migraphx::gpu::hip_compile_options options;
    options.inputs      = {packed, packed, accum, output};
    options.output      = output;
    options.output_arg  = 3;
    options.local       = 256;
    options.global      = ((left.size() + 255) / 256) * 256;
    options.kernel_name = "packed_math";
    const auto source   = R"__migraphx__(
#include <migraphx/kernels/packed_conv.hpp>
extern "C" __global__ void packed_math(const migraphx::uint32_t* a, const migraphx::uint32_t* b,
                                      const float* c, float* y)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= 864) return;
    y[4*i]   = migraphx::packed_conv_mad_mix<0,0>(a[i],b[i],c[i]);
    y[4*i+1] = migraphx::packed_conv_mad_mix<0,1>(a[i],b[i],c[i]);
    y[4*i+2] = migraphx::packed_conv_mad_mix<1,0>(a[i],b[i],c[i]);
    y[4*i+3] = migraphx::packed_conv_mad_mix<1,1>(a[i],b[i],c[i]);
}
)__migraphx__";
    const auto kernel   = migraphx::gpu::compile_hip_code_object(ctx, source, options);
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::instruction_ref> args;
    for(std::size_t i = 0; i < options.inputs.size(); ++i)
        args.push_back(mm->add_parameter("x" + std::to_string(i), options.inputs[i]));
    mm->add_return({mm->add_instruction(kernel, args)});
    p.compile(migraphx::make_target("gpu"), {});
    const auto result = migraphx::gpu::from_gpu(
        p.eval({{"x0", migraphx::gpu::to_gpu(migraphx::argument{packed, left.data()})},
                {"x1", migraphx::gpu::to_gpu(migraphx::argument{packed, right.data()})},
                {"x2", migraphx::gpu::to_gpu(migraphx::argument{accum, sums.data()})},
                {"x3", migraphx::gpu::allocate_gpu(output)}})
            .front());
    result.visit([&](auto values) {
        for(std::size_t i = 0; i < expected.size(); ++i)
            EXPECT(float(values[i]) == expected[i]);
    });
}

TEST_CASE(packed_conv_large_numerics) { check_native(512, false); }
TEST_CASE(packed_conv_small_numerics) { check_native(32, false); }
TEST_CASE(packed_conv_fused_numerics) { check_native(32, true); }
TEST_CASE(packed_conv_permuted_arguments) { check_native(32, false, true); }
TEST_CASE(packed_conv_permuted_fused_arguments) { check_native(32, true, true); }
TEST_CASE(packed_conv_paired_compiler) { check_native(32, true, true, true); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
