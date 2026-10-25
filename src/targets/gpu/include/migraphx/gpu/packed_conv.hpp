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
#ifndef MIGRAPHX_GUARD_GPU_PACKED_CONV_HPP
#define MIGRAPHX_GUARD_GPU_PACKED_CONV_HPP

#include <migraphx/gpu/compiler.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct packed_conv_problem
{
    std::size_t input_channels;
    bool pointwise;
    std::vector<std::size_t> arguments;
    instruction_ref activation;
};

MIGRAPHX_GPU_EXPORT optional<packed_conv_problem> analyze_packed_conv(
    const module& m, const std::vector<shape>& inputs, std::string* rejection = nullptr);
MIGRAPHX_GPU_EXPORT bool packed_conv_supported(const context& ctx);
MIGRAPHX_GPU_EXPORT std::vector<value> packed_conv_solutions(const packed_conv_problem& problem,
                                                             bool exhaustive);
MIGRAPHX_GPU_EXPORT optional<tuning_config> get_packed_conv_tuning_config(context& ctx,
                                                                          instruction_ref ins,
                                                                          bool exhaustive,
                                                                          bool fast_math = false);
MIGRAPHX_GPU_EXPORT operation compile_packed_conv(context& ctx,
                                                  const module& m,
                                                  const std::vector<shape>& inputs,
                                                  const value& solution);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
