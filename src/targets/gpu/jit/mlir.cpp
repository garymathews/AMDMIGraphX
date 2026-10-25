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

#include <iterator>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>
#include <migraphx/gpu/packed_conv.hpp>
#include <migraphx/program.hpp>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_DUMP_TO_MXR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_DUMP);

static module create_pointwise_module(module_ref in_mod)
{
    module pw_mod;
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    for(auto param : in_mod->get_parameters())
    {
        map_ins[param] =
            pw_mod.add_parameter(any_cast<builtin::param>(param->get_operator()).parameter,
                                 shape{param->get_shape().type()});
    }
    auto return_args =
        pw_mod.add_instructions(in_mod,
                                &map_ins,
                                [](module& m,
                                   instruction_ref ins,
                                   const operation& op,
                                   const std::vector<instruction_ref>& inputs,
                                   const std::vector<module_ref>& mod_args) -> instruction_ref {
                                    auto out_aliases = op.output_alias(to_shapes(inputs));
                                    if(out_aliases.size() == 1)
                                        return inputs.at(out_aliases[0]);
                                    else
                                        return m.insert_instruction(ins, op, inputs, mod_args);
                                });
    pw_mod.add_return(return_args);
    return pw_mod;
}

static code_object_op
compile_pointwise_module(context& ctx, const std::vector<shape>& inputs, module_ref mod)
{
    operation cop;
    auto pw_mod = create_pointwise_module(mod);
    if(any_of(mod->get_parameters(), [&](instruction_ref param) {
           if(param->outputs().size() != 1)
               return false;
           return equal(instruction::get_output_alias(param->outputs().front(), /* shallow */ true),
                        {param});
       }))
    {
        auto mod2 = *mod;
        adjust_param_shapes(mod2, inputs);
        auto names = mod2.get_parameter_names();
        std::sort(names.begin(), names.end());
        std::vector<shape> new_shapes;
        std::transform(names.begin(),
                       names.end(),
                       std::back_inserter(new_shapes),
                       [&](const std::string& name) {
                           auto param       = mod2.get_parameter(name);
                           auto output_path = get_output_path(param);
                           auto it          = std::adjacent_find(
                               output_path.begin(),
                               output_path.end(),
                               [&](instruction_ref, instruction_ref output) {
                                   return not equal(instruction::get_output_alias(output), {param});
                               });
                           return (*it)->get_shape();
                       });
        std::copy(inputs.begin() + new_shapes.size(), inputs.end(), std::back_inserter(new_shapes));
        cop = compile_pointwise(ctx, new_shapes, &pw_mod);
    }
    else
    {
        cop = compile_pointwise(ctx, inputs, &pw_mod);
    }
    auto co            = any_cast<code_object_op>(cop);
    co.expected_inputs = inputs;
    return co;
}

static instruction_ref find_final_split(instruction_ref split_ins)
{
    auto output_path = get_output_path(split_ins);
    auto it          = std::adjacent_find(
        output_path.begin(), output_path.end(), [&](instruction_ref input, instruction_ref output) {
            if(contains({"reshape", "squeeze", "unsqueeze", "transpose"}, output->name()))
                return false;
            if(contains({"add", "mul"}, output->name()))
            {
                auto aux = *std::find_if(output->inputs().begin(),
                                         output->inputs().end(),
                                         [&](instruction_ref i) { return i != input; });
                if(aux->can_eval())
                    return false;
                auto aliases = instruction::get_output_alias(aux);
                return aliases.size() == 1 and aliases[0]->name() != "@param";
            }
            return true;
        });
    return *it;
}

instruction_ref insert_mlir(module& m,
                            instruction_ref ins,
                            const mlir_code_object& compiled,
                            std::vector<instruction_ref> inputs)
{
    // Kernels receive flattened arguments, but retain the output tuple's alias.
    assert(std::all_of(inputs.begin(), inputs.end() - 1, [](auto input) {
        return input->get_shape().sub_shapes().empty();
    }));
    const auto output_index = inputs.size() - 1;
    const bool multi_output = not inputs.back()->get_shape().sub_shapes().empty();
    std::vector<instruction_ref> output_fills;
    for(auto i : range(compiled.prefill_indices.size()))
    {
        const auto index         = compiled.prefill_indices[i];
        const bool tuple_element = multi_output and index >= output_index;
        auto buffer =
            tuple_element
                ? m.insert_instruction(ins,
                                       make_op("get_tuple_elem", {{"index", index - output_index}}),
                                       inputs.back())
                : inputs.at(index);
        auto filled = m.insert_instruction(
            ins, make_op("hip::fill", {{"value", compiled.prefill_values.at(i)}}), buffer);
        if(tuple_element)
            output_fills.push_back(filled);
        else
            replace(inputs, buffer, filled);
    }
    if(not output_fills.empty())
    {
        // Preserve the tuple and make the kernel depend on every output fill.
        output_fills.insert(output_fills.begin(), inputs.back());
        inputs.back() = m.insert_instruction(ins, make_op("identity"), output_fills);
    }
    return insert_mlir(m, ins, compiled.cop, inputs);
}

struct mlir_compiler : compiler<mlir_compiler>
{
    std::vector<std::string> names() const { return {"gpu::mlir_op"}; }

    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }
    std::optional<instruction_ref> input_is_param(const instruction_ref& ins) const
    {
        auto aliases = instruction::get_output_alias(ins);
        for(auto cur : aliases)
        {
            while(contains({"reshape", "contiguous"}, cur->name()))
            {
                auto nested_aliases = instruction::get_output_alias(cur->inputs().at(0));
                cur                 = nested_aliases.front();
            }
            if(cur->name() == "@param")
            {
                return cur;
            }
        }
        return nullopt;
    }

    bool is_range_literal(const instruction_ref& ins) const
    {
        auto lit_elms = ins->get_shape().element_space();
        if(not ins->can_eval() or lit_elms < 2)
        {
            return false;
        }
        bool is_range = false;
        ins->eval().visit([&](auto l) {
            is_range = std::adjacent_find(l.begin(), l.begin() + lit_elms, [](auto cur, auto next) {
                           return not float_equal(next - cur, 1.0);
                       }) == l.begin() + lit_elms;
        });
        return is_range;
    }

    void set_fill_map(compiler_replace& cr, const module& m) const
    {
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "greater")
            {
                continue;
            }
            auto fill_val      = ins->get_shape().lens().back() - 1;
            bool has_range_lit = std::any_of(ins->inputs().begin(),
                                             ins->inputs().end(),
                                             [&](auto inp) { return is_range_literal(inp); });
            for(auto inp : ins->inputs())
            {
                auto param = input_is_param(inp);
                if(param.has_value() and has_range_lit)
                {
                    auto id = param.value()->get_shape().type_string() +
                              migraphx::shape::to_sizes_string({param.value()->get_shape()});
                    cr.fill_map[id] = static_cast<double>(fill_val);
                }
            }
        }
    }

    compiler_replace compile(context& ctx,
                             instruction_ref ins,
                             const operation& op,
                             const value& configuration) const
    {
        const auto provider = configuration.is_object() ? configuration.get("provider", "") : "";
        if(contains({"packed_conv", "rocmlir_pair"}, provider) and
           not op.to_value().get("fast_math", false))
            MIGRAPHX_THROW("Packed convolution requires fast_math FP32 fusion permission");
        if(provider == "packed_conv")
            return compile_packed_conv(
                ctx, *ins->module_inputs().front(), to_shapes(ins->inputs()), configuration);
        if(provider == "rocmlir_pair")
            return compile_head_pair(ctx, ins, configuration);
        const auto& solution = provider == "rocmlir" ? configuration.at("config") : configuration;
        auto* smod           = ins->module_inputs().front();
        assert(smod->get_parameter_names().size() == ins->inputs().size() - 1);
        auto gemm_like_ins = std::find_if(smod->begin(), smod->end(), [&](const auto& i) {
            return contains({"dot", "quant_dot", "convolution", "quant_convolution"}, i.name());
        });
        auto pointwise_ins = std::find_if(gemm_like_ins, smod->end(), [&](const auto& i) {
            return i.get_operator().attributes().get("pointwise", false) == true;
        });

        // check if (a) module is fused (b) contains a "gemm/conv" instruction and (c)
        // perfConfig can not allow fused module
        if(gemm_like_ins != smod->end() and pointwise_ins != smod->end() and
           solution.if_string() != nullptr and not is_module_fusible(*smod, ctx, solution))
        {
            auto input_args = ins->inputs();
            // remove alloc buffer
            input_args.pop_back();
            auto split_ins                               = find_final_split(gemm_like_ins);
            std::array<module_with_inputs, 2> mod_splits = smod->split(input_args, {split_ins});
            auto dot_mlir_inputs = to_shapes(mod_splits[0].inputs);
            // add alloc for the gemm output
            dot_mlir_inputs.push_back(mod_splits[0].mod.get_output_shapes().front());
            mlir_code_object cop1 = compile_mlir(ctx, mod_splits[0].mod, dot_mlir_inputs, solution);
            auto pw_shapes        = to_shapes(mod_splits[1].inputs);
            if(mod_splits[1].mod.get_output_shapes().size() == 1)
            {
                pw_shapes.push_back(mod_splits[1].mod.get_output_shapes().front());
            }
            else
            {
                pw_shapes.push_back(shape{mod_splits[1].mod.get_output_shapes()});
            }
            assert(pw_shapes.back() == ins->get_shape());
            auto cop2 = compile_pointwise_module(ctx, pw_shapes, &mod_splits[1].mod);
            std::array<compiler_replace, 2> cops = {insert(cop1), insert(mlir_code_object{cop2})};
            return insert(cops, mod_splits, ins, split_ins);
        }
        auto cr = insert(compile_mlir(ctx, *smod, to_shapes(ins->inputs()), solution));
        set_fill_map(cr, *smod);
        return cr;
    }

    compiler_replace
    compile_head_pair(context& ctx, instruction_ref ins, const value& solution) const
    {
        auto* smod         = ins->module_inputs().front();
        const auto problem = analyze_packed_conv(*smod, to_shapes(ins->inputs()));
        if(not problem or not problem->pointwise)
            MIGRAPHX_THROW("Unsupported two-stage packed convolution fallback");
        auto inputs = ins->inputs();
        inputs.pop_back();
        auto mods    = smod->split(inputs, {problem->activation});
        auto staging = std::make_shared<program>();
        std::array<compiler_replace, 2> kernels;
        for(std::size_t i = 0; i < mods.size(); ++i)
        {
            auto shapes = to_shapes(mods[i].inputs);
            shapes.push_back(mods[i].mod.get_output_shapes().front());
            auto* body = staging->create_module("stage" + std::to_string(i), mods[i].mod);
            auto* mm   = staging->get_main_module();
            std::vector<instruction_ref> args;
            for(std::size_t j = 0; j < shapes.size(); ++j)
                args.push_back(mm->add_parameter(
                    "stage" + std::to_string(i) + "_x" + std::to_string(j), shapes[j]));
            auto stage = mm->add_instruction(
                make_op("gpu::precompile_op", {{"output_shape", to_value(shapes.back())}}),
                args,
                {body});
            kernels[i] = compile(ctx,
                                 stage,
                                 make_op("gpu::mlir_op", {{"fast_math", true}}),
                                 solution.at(i == 0 ? "first" : "second"));
        }
        return insert(kernels, mods, ins, problem->activation, staging);
    }

    compiler_replace insert(const mlir_code_object& mco) const
    {
        return {std::vector<operation>{mco.cop},
                [=](module& m, instruction_ref ins, const std::vector<operation>& ops) {
                    auto compiled = mco;
                    compiled.cop  = any_cast<code_object_op>(ops.front());
                    return m.replace_instruction(ins, insert_mlir(m, ins, compiled, ins->inputs()));
                },
                &trace};
    }

    compiler_replace insert(const std::array<compiler_replace, 2>& stages,
                            const std::array<module_with_inputs, 2>& mods,
                            instruction_ref precompile_ins,
                            instruction_ref split_ins,
                            std::shared_ptr<program> lifetime = nullptr) const
    {
        std::vector<operation> objects;
        for(const auto& stage : stages)
            objects.insert(objects.end(), stage.code_objects.begin(), stage.code_objects.end());
        auto original_inputs = precompile_ins->inputs();
        return {
            objects, [=](module& m, instruction_ref ins, const std::vector<operation>& ops) {
                // Stage compilers may retain references to their source program.
                [[maybe_unused]] const auto keep_alive = lifetime;
                std::unordered_map<instruction_ref, instruction_ref> replacements;
                for(auto i : range(original_inputs.size()))
                    replacements[original_inputs[i]] = ins->inputs()[i];
                const auto remap = [&](std::vector<instruction_ref> inputs) {
                    for(auto& input : inputs)
                        if(auto found = replacements.find(input); found != replacements.end())
                            input = found->second;
                    return inputs;
                };
                auto first_inputs      = remap(mods[0].inputs);
                const auto first_shape = mods[0].mod.get_output_shapes().front();
                first_inputs.push_back(m.insert_instruction(
                    ins, make_op("hip::allocate", {{"shape", to_value(first_shape)}})));
                auto first = m.insert_instruction(
                    ins,
                    make_op("gpu::precompile_op", {{"output_shape", to_value(first_shape)}}),
                    first_inputs);
                auto second_module = mods[1];
                second_module.replace(split_ins, first);
                auto second_inputs = remap(second_module.inputs);
                second_inputs.push_back(ins->inputs().back());
                auto second = m.insert_instruction(
                    ins,
                    make_op("gpu::precompile_op", {{"output_shape", to_value(ins->get_shape())}}),
                    second_inputs);
                // Keep real consumers attached while each compiler rewires its stage.
                m.replace_instruction(ins, make_op("identity"), {second});
                std::size_t offset = 0;
                for(auto i : range(stages.size()))
                {
                    auto compiled = stages[i];
                    compiled.code_objects.assign(
                        ops.begin() + offset, ops.begin() + offset + compiled.code_objects.size());
                    offset += compiled.code_objects.size();
                    compiled.replace(m, i == 0 ? first : second);
                }
                return m.replace_instruction(ins, ins->inputs().front());
            }};
    }

    optional<tuning_config>
    get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    {
        static const auto mxr_loc  = string_value_of(MIGRAPHX_MLIR_DUMP_TO_MXR{});
        static const auto mlir_loc = string_value_of(MIGRAPHX_MLIR_DUMP{});

        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        if(not mxr_loc.empty())
        {
            dump_mlir_to_mxr(*smod, ins->inputs(), mxr_loc);
        }
        if(not mlir_loc.empty())
        {
            dump_mlir_to_file(*smod, shapes, mlir_loc);
        }
        if(auto packed = get_packed_conv_tuning_config(
               ctx, ins, exhaustive, op.to_value().get("fast_math", false)))
            return packed;
        return get_tuning_config_mlir(ctx, *smod, shapes, exhaustive);
    }

    static void trace(std::ostream& os, instruction_ref ins)
    {
        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        os << dump_mlir(*smod, shapes);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
