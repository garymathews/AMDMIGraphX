/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/json.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_TUNE_EXHAUSTIVE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_TUNE_LIMIT);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_HIP_FLAGS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_OPTIMIZE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_MLIR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_BENCHMARKING);
namespace {

instruction_ref unwrap(instruction_ref ins)
{
    while(contains({"reshape", "contiguous"}, ins->name()) and ins->get_shape().standard() and
          ins->inputs().front()->get_shape().standard() and
          ins->get_shape().type() == ins->inputs().front()->get_shape().type())
        ins = ins->inputs().front();
    return ins;
}

struct biased_conv
{
    instruction_ref conv;
    instruction_ref bias;
};

optional<biased_conv> match_biased_conv(instruction_ref ins, std::size_t kernel)
{
    ins = unwrap(ins);
    if(ins->name() != "add" or ins->inputs().size() != 2)
        return nullopt;
    auto conv = ins->inputs()[0];
    auto bias = ins->inputs()[1];
    if(conv->name() != "convolution")
        std::swap(conv, bias);
    if(conv->name() != "convolution" or conv->inputs().size() != 2 or
       not contains({"broadcast", "multibroadcast"}, bias->name()))
        return nullopt;
    const auto& w = conv->inputs()[1]->get_shape();
    const auto& b = bias->get_shape();
    if(w.ndim() != 4 or w.lens()[2] != kernel or w.lens()[3] != kernel or
       b.lens() != conv->get_shape().lens() or b.strides() != std::vector<std::size_t>{0, 1, 0, 0})
        return nullopt;
    const auto attrs   = conv->get_operator().to_value();
    const auto padding = attrs.at("padding").to_vector<std::size_t>();
    if(attrs.at("group").to<std::size_t>() != 1 or
       attrs.at("stride").to_vector<std::size_t>() != std::vector<std::size_t>{1, 1} or
       attrs.at("dilation").to_vector<std::size_t>() != std::vector<std::size_t>{1, 1} or
       not(padding.size() == 2 or padding.size() == 4) or
       not std::all_of(padding.begin(), padding.end(), [&](auto p) { return p == kernel / 2; }))
        return nullopt;
    return biased_conv{conv, unwrap(bias->inputs().front())};
}

instruction_ref match_silu(instruction_ref ins)
{
    ins = unwrap(ins);
    if(ins->name() != "mul" or ins->inputs().size() != 2)
        return ins;
    auto x       = ins->inputs()[0];
    auto sigmoid = ins->inputs()[1];
    if(sigmoid->name() != "sigmoid")
        std::swap(x, sigmoid);
    if(sigmoid->name() == "sigmoid" and sigmoid->inputs().front() == x)
        return x;
    return ins;
}

value packed_tuning_key(const context& ctx,
                        const value& base,
                        const std::vector<shape>& inputs,
                        bool exhaustive)
{
    const auto& device = ctx.get_current_device();
    return to_json_string(value{{"packed_conv_version", 9},
                                {"compiler", MIGRAPHX_VERSION_TWEAK},
                                {"math", "fp32_fusion"},
                                {"exhaustive", exhaustive},
                                {"mlir_tune_limit", string_value_of(MIGRAPHX_MLIR_TUNE_LIMIT{})},
                                {"jit_flags", string_value_of(MIGRAPHX_GPU_HIP_FLAGS{})},
                                {"jit_optimize", string_value_of(MIGRAPHX_GPU_OPTIMIZE{}, "3")},
                                {"base", base},
                                {"arch", device.get_device_name()},
                                {"cu", device.get_cu_count()},
                                {"inputs", to_value(inputs)}});
}

// Select each ordinary stage once, then compare whole-region alternatives. The
// existing compiler applies fills and allocations to the benchmark as at runtime.
std::pair<value, value> select_mlir_stage(context& ctx,
                                          const module& stage,
                                          const std::vector<shape>& shapes,
                                          bool exhaustive)
{
    auto tc         = get_tuning_config_mlir(ctx, stage, shapes, exhaustive);
    const auto seed = ctx.get_problem_cache().get("gpu::mlir_op", tc.problem).value_or(value{});
    const auto key =
        packed_tuning_key(ctx, value{{"stage", tc.problem}, {"seed", seed}}, shapes, exhaustive);
    if(auto saved = ctx.get_problem_cache().get("gpu::mlir_op", key))
        return {key, saved->at("config")};
    if(not seed.is_null() and not contains(tc.solutions, seed))
        tc.solutions.insert(tc.solutions.begin(), seed);
    tc.solutions.push_back(value{});
    double best = std::numeric_limits<double>::infinity();
    value winner;
    for(const auto& solution : tc.solutions)
    {
        try
        {
            if(value_of(MIGRAPHX_TRACE_BENCHMARKING{}) > 1)
                std::cout << "packed_conv stage compiling: " << solution << std::endl;
            program bench;
            auto* body = bench.create_module("stage", stage);
            auto* mm   = bench.get_main_module();
            std::vector<instruction_ref> args;
            for(std::size_t i = 0; i < shapes.size(); ++i)
                args.push_back(mm->add_parameter("x" + std::to_string(i), shapes[i]));
            auto ins = mm->add_instruction(
                make_op("gpu::precompile_op", {{"output_shape", to_value(shapes.back())}}),
                args,
                {body});
            mm->add_return({ins});
            gpu::compile(ctx, ins, make_op("gpu::mlir_op"), solution).replace(*mm, ins);
            run_passes(
                *mm,
                {eliminate_identity{}, dead_code_elimination{}, memory_coloring{"hip::allocate"}});
            const auto elapsed = time_program(ctx, std::move(bench), {}, 10, 20);
            if(value_of(MIGRAPHX_TRACE_BENCHMARKING{}) > 1)
                std::cout << "packed_conv stage: " << solution << " " << elapsed << "ms"
                          << std::endl;
            if(elapsed < best)
            {
                best   = elapsed;
                winner = solution;
            }
        }
        catch(const std::exception& e)
        {
            if(value_of(MIGRAPHX_TRACE_BENCHMARKING{}) > 0)
                std::cout << "packed_conv stage rejected: " << e.what() << std::endl;
        }
    }
    if(not std::isfinite(best))
        MIGRAPHX_THROW("No valid ordinary packed-head stage configuration");
    ctx.get_problem_cache().insert("gpu::mlir_op", key, value{{"config", winner}});
    return {key, winner};
}

} // namespace

optional<packed_conv_problem>
analyze_packed_conv(const module& m, const std::vector<shape>& inputs, std::string* rejection)
{
    const auto reject = [&](const char* reason) -> optional<packed_conv_problem> {
        if(rejection != nullptr)
            *rejection = reason;
        return nullopt;
    };
    auto params = m.get_parameters();
    sort_params(params);
    if((params.size() != 3 and params.size() != 5) or
       (inputs.size() != params.size() and inputs.size() != params.size() + 1))
        return reject("parameter count");
    const auto outputs = m.get_output_shapes();
    if(outputs.size() != 1 or outputs.front().type() != shape::half_type or
       outputs.front().dynamic() or not outputs.front().standard())
        return reject("expected one static contiguous FP16 output");
    for(std::size_t i = 0; i < params.size(); ++i)
        if(inputs[i] != params[i]->get_shape() or inputs[i].type() != shape::half_type or
           inputs[i].dynamic() or not inputs[i].standard())
            return reject("expected static contiguous FP16 inputs");
    if(inputs.size() == params.size() + 1 and inputs.back() != outputs.front())
        return reject("output allocation shape");
    for(auto ins : iterator_for(m))
        if(not contains({"@param",
                         "@return",
                         "convolution",
                         "broadcast",
                         "multibroadcast",
                         "add",
                         "sigmoid",
                         "mul",
                         "reshape",
                         "contiguous"},
                        ins->name()))
            return reject("unsupported operation");
    auto result = std::prev(m.end())->inputs();
    if(result.size() != 1)
        return reject("multiple return values");
    auto activation            = unwrap(result.front());
    const bool pointwise       = params.size() == 5;
    optional<biased_conv> tail = nullopt;
    if(pointwise)
    {
        tail = match_biased_conv(activation, 1);
        if(not tail or tail->conv->get_shape() != shape{shape::half_type, {1, 4, 10, 10}} or
           tail->conv->inputs()[1]->get_shape() != shape{shape::half_type, {4, 32, 1, 1}})
            return reject("unsupported pointwise convolution");
        activation = tail->conv->inputs().front();
    }
    auto preactivation = match_silu(activation);
    if(preactivation == unwrap(activation))
        return reject("expected SiLU");
    auto head = match_biased_conv(preactivation, 3);
    if(not head or head->conv->get_shape() != shape{shape::half_type, {1, 32, 10, 10}})
        return reject("unsupported biased3x3 convolution");
    const auto x = head->conv->inputs()[0];
    const auto w = head->conv->inputs()[1];
    if(x->get_shape().ndim() != 4)
        return reject("expected NCHW input");
    const auto ic = x->get_shape().lens()[1];
    if((ic != 32 and ic != 512) or (pointwise and ic != 32) or
       x->get_shape() != shape{shape::half_type, {1, ic, 10, 10}} or
       w->get_shape() != shape{shape::half_type, {32, ic, 3, 3}} or
       head->bias->get_shape() != shape{shape::half_type, {32}})
        return reject("unsupported convolution shape");
    std::vector<instruction_ref> ordered{x, w, head->bias};
    if(pointwise)
    {
        if(tail->bias->get_shape() != shape{shape::half_type, {4}})
            return reject("pointwise bias shape");
        ordered.push_back(tail->conv->inputs()[1]);
        ordered.push_back(tail->bias);
    }
    packed_conv_problem p{ic, pointwise, {}, activation};
    for(auto arg : ordered)
    {
        const auto found = std::find(params.begin(), params.end(), arg);
        if(found == params.end())
            return reject("non-parameter operand");
        p.arguments.push_back(std::distance(params.begin(), found));
    }
    if(std::unordered_set<std::size_t>(p.arguments.begin(), p.arguments.end()).size() !=
       params.size())
        return reject("aliased operands");
    return p;
}

bool packed_conv_supported(const context& ctx)
{
    const auto& d = ctx.get_current_device();
    return contains({"gfx900", "gfx90c"}, d.get_gfx_name()) and d.get_wavefront_size() == 64 and
           d.get_max_workitems_per_block() >= 512;
}

std::vector<value> packed_conv_solutions(const packed_conv_problem& p, bool exhaustive)
{
    std::vector<value> solutions;
    std::vector<std::array<std::size_t, 4>> configurations =
        p.input_channels == 512
            ? std::vector<std::array<std::size_t, 4>>{{32, 8, 16, 8}, {16, 16, 8, 8}}
            : std::vector<std::array<std::size_t, 4>>{
                  {32, 8, 16, 6}, {32, 8, 36, 8}, {32, 8, 8, 4}};
    if(exhaustive)
    {
        configurations.clear();
        const auto rows =
            p.pointwise ? std::vector<std::size_t>{32} : std::vector<std::size_t>{16, 32};
        for(auto tm : rows)
            for(auto tn : {8u, 16u, 32u})
                for(auto tk : {8u, 16u, 18u, 32u, 36u})
                    for(auto split : {1u, 2u, 3u, 4u, 6u, 8u})
                        configurations.push_back({tm, tn, tk, split});
    }
    for(const auto& [tm, tn, tk, split] : configurations)
    {
        const auto operands = 4 * split * (tm * (tk / 2 + 1) + tk * (tn / 2 + 1));
        const auto partials = 4 * (split > 1 ? split - 1 : 1) * tm * (tn + 1);
        const auto lds      = std::max(operands, partials) + (p.pointwise ? 2 * tm * tn : 0);
        if(lds <= 65536 and (p.input_channels * 9) % (tk * split) == 0)
            solutions.push_back(value{
                {"provider", "packed_conv"}, {"tm", tm}, {"tn", tn}, {"tk", tk}, {"split", split}});
    }
    return solutions;
}

optional<tuning_config>
get_packed_conv_tuning_config(context& ctx, instruction_ref ins, bool exhaustive, bool fast_math)
{
    if(not fast_math or not mlir_enabled())
        return nullopt;
    exhaustive        = exhaustive or enabled(MIGRAPHX_MLIR_TUNE_EXHAUSTIVE{});
    auto* smod        = ins->module_inputs().front();
    const auto shapes = to_shapes(ins->inputs());
    std::string rejection;
    const auto p = analyze_packed_conv(*smod, shapes, &rejection);
    if(not p or not packed_conv_supported(ctx))
    {
        if(enabled(MIGRAPHX_TRACE_MLIR{}))
            std::cout << "packed_conv rejected: " << (p ? "unsupported device" : rejection)
                      << std::endl;
        return nullopt;
    }
    tuning_config tc;
    if(p->pointwise)
    {
        auto inputs = ins->inputs();
        inputs.pop_back();
        const auto mods = smod->split(inputs, {p->activation});
        value fallback{{"provider", "rocmlir_pair"}, {"first", value{}}, {"second", value{}}};
        value problems = value::array{};
        for(std::size_t i = 0; i < mods.size(); ++i)
        {
            auto stage_shapes = to_shapes(mods[i].inputs);
            stage_shapes.push_back(mods[i].mod.get_output_shapes().front());
            const auto selected = select_mlir_stage(ctx, mods[i].mod, stage_shapes, exhaustive);
            problems.push_back(value{{"problem", selected.first}, {"config", selected.second}});
            fallback[i == 0 ? "first" : "second"] = selected.second;
        }
        tc.problem              = problems;
        tc.solutions            = {fallback};
        const auto stage_shapes = to_shapes(mods[0].inputs);
        if(auto first = analyze_packed_conv(mods[0].mod, stage_shapes))
            for(const auto& native : packed_conv_solutions(*first, exhaustive))
            {
                auto pair     = fallback;
                pair["first"] = native;
                tc.solutions.push_back(pair);
            }
    }
    else
    {
        tc = get_tuning_config_mlir(ctx, *smod, shapes, exhaustive);
        if(not exhaustive)
            if(auto cached = ctx.get_problem_cache().get("gpu::mlir_op", tc.problem))
                tc.solutions = {*cached};
        for(auto& solution : tc.solutions)
            solution = value{{"provider", "rocmlir"}, {"config", solution}};
    }
    // Include selected stage configurations, not only their shape-based problem
    // keys, so changing a stage winner invalidates dependent region selection.
    tc.problem =
        packed_tuning_key(ctx,
                          value{{"problem", tc.problem}, {"arguments", to_value(p->arguments)}},
                          shapes,
                          exhaustive);
    if(value_of(MIGRAPHX_TRACE_BENCHMARKING{}) > 0)
        if(auto cached = ctx.get_problem_cache().get("gpu::mlir_op", tc.problem))
            std::cout << "packed_conv cache hit: " << *cached << std::endl;
    auto native = packed_conv_solutions(*p, exhaustive);
    tc.solutions.insert(tc.solutions.end(), native.begin(), native.end());
    return tc;
}

operation compile_packed_conv(context& ctx,
                              const module& m,
                              const std::vector<shape>& shapes,
                              const value& solution)
{
    const auto p = analyze_packed_conv(m, shapes);
    if(not p or shapes.size() != p->arguments.size() + 1 or not packed_conv_supported(ctx))
        MIGRAPHX_THROW("Unsupported packed convolution");
    const auto valid = packed_conv_solutions(*p, true);
    const std::array<std::string, 4> fields{"tm", "tn", "tk", "split"};
    const auto supported = std::any_of(valid.begin(), valid.end(), [&](const auto& candidate) {
        return std::all_of(fields.begin(), fields.end(), [&](const auto& name) {
            return candidate.at(name).template to<std::size_t>() ==
                   solution.at(name).template to<std::size_t>();
        });
    });
    if(not supported)
        MIGRAPHX_THROW("Invalid packed convolution configuration");
    hip_compile_options options;
    options.inputs      = shapes;
    options.output      = shapes.back();
    options.output_arg  = options.inputs.size() - 1;
    options.kernel_name = p->pointwise ? "packed_conv_pointwise" : "packed_conv_silu";
    const auto tm       = solution.at("tm").to<std::size_t>();
    const auto tn       = solution.at("tn").to<std::size_t>();
    const auto split    = solution.at("split").to<std::size_t>();
    options.local       = 64 * split;
    options.global      = ((32 + tm - 1) / tm) * ((100 + tn - 1) / tn) * options.local;
    std::string parameters;
    for(std::size_t i = 0; i < p->arguments.size(); ++i)
        parameters += "const half* x" + std::to_string(i) + ", ";
    std::string arguments;
    for(auto index : p->arguments)
        arguments += "x" + std::to_string(index) + ", ";
    if(not p->pointwise)
        arguments += "nullptr, nullptr, ";
    const auto source =
        interpolate_string(R"__migraphx__(
#include <migraphx/kernels/packed_conv.hpp>
namespace migraphx {
extern "C" __global__ __launch_bounds__(${local})
void ${kernel}(${parameters}half* y)
{
    migraphx::packed_conv<${ic}, ${tm}, ${tn}, ${tk}, ${split}, ${pointwise}>(
        ${arguments}y);
}
} // namespace migraphx
)__migraphx__",
                           {{"local", std::to_string(options.local)},
                            {"kernel", options.kernel_name},
                            {"parameters", parameters},
                            {"arguments", arguments},
                            {"ic", std::to_string(p->input_channels)},
                            {"tm", std::to_string(tm)},
                            {"tn", std::to_string(tn)},
                            {"tk", std::to_string(solution.at("tk").to<std::size_t>())},
                            {"split", std::to_string(split)},
                            {"pointwise", p->pointwise ? "true" : "false"}});
    return compile_hip_code_object(ctx, source, options);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
